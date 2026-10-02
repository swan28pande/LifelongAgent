"""Real provider wrappers with mocked transports; no credentials or network needed."""

import json

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from memory_v4.retrieval import PlannedMemoryReader


def decision(action="retrieve", text=""):
    return {
        "action": action,
        "evidence_gap": "Need the stored value" if action == "retrieve" else "",
        "tool_name": "lookup" if action == "retrieve" else "",
        "arguments_json": '{"topic":"coffee"}' if action == "retrieve" else "{}",
        "answer": text,
    }


@pytest.fixture(params=["openai", "gemini"])
def provider(request, monkeypatch):
    requests = []
    responses = []
    if request.param == "openai":
        integration = pytest.importorskip("langchain_openai")
        import httpx

        def respond(http_request):
            requests.append(json.loads(http_request.content))
            payload, native_call = responses.pop(0)
            message = {"role": "assistant", "content": payload}
            if native_call:
                message["tool_calls"] = [
                    {
                        "id": "unplanned",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"topic":"wrong"}',
                        },
                    }
                ]
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(respond))
        llm = integration.ChatOpenAI(
            model="gpt-4.1-mini",
            api_key="test",
            http_client=client,
            timeout=60,
            max_retries=2,
        )
        close = client.close
    else:
        integration = pytest.importorskip("langchain_google_genai")
        from google.genai.types import (
            Candidate,
            Content,
            FunctionCall,
            GenerateContentResponse,
            Part,
        )

        llm = integration.ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            api_key="test",
            vertexai=False,
            timeout=60,
            max_retries=2,
        )

        def respond(**sdk_request):
            requests.append(sdk_request)
            payload, native_call = responses.pop(0)
            parts = [Part(text=payload)]
            if native_call:
                parts.append(
                    Part(
                        function_call=FunctionCall(
                            name="lookup", args={"topic": "wrong"}
                        )
                    )
                )
            return GenerateContentResponse(
                candidates=[
                    Candidate(
                        content=Content(role="model", parts=parts), finish_reason="STOP"
                    )
                ],
                usage_metadata={
                    "prompt_token_count": 5,
                    "candidates_token_count": 3,
                    "total_token_count": 8,
                },
            )

        monkeypatch.setattr(llm.client.models, "generate_content", respond)
        close = llm.client.close

    yield request.param, llm, responses, requests
    close()


class CaptureCallbacks(BaseCallbackHandler):
    def __init__(self):
        self.usage = []
        self.tools = []

    def on_llm_end(self, response, **kwargs):
        self.usage.append(response.generations[0][0].message.usage_metadata)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tools.append(serialized["name"])


def reader_for(llm, calls):
    @tool
    def lookup(topic: str) -> str:
        """Retrieve one stored value."""
        calls.append(topic)
        return "tea"

    return PlannedMemoryReader(llm, [lookup])


def assert_schema_request(provider_name, request, title):
    if provider_name == "openai":
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["name"] == title
        assert "tools" not in request
    else:
        config = request["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema["title"] == title
        assert config.tools is None and config.http_options.timeout == 60_000


def test_native_schema_reassessment_and_callbacks(provider):
    name, llm, responses, requests = provider
    responses.extend(
        [
            (json.dumps(decision()), False),
            (json.dumps(decision("answer", "tea")), False),
        ]
    )
    calls = []
    callbacks = CaptureCallbacks()
    result = reader_for(llm, calls).invoke(
        {"messages": [HumanMessage(content="What do I drink?")]},
        config={"callbacks": [callbacks]},
    )
    assert result["answer"] == "tea" and result["num_tool_calls"] == len(calls) == 1
    assert (
        result["rejections"] == [] and result["num_model_calls"] == len(requests) == 2
    )
    assert callbacks.tools == ["lookup"]
    assert [usage["total_tokens"] for usage in callbacks.usage] == [8, 8]
    for sent in requests:
        assert_schema_request(name, sent, "RetrievalDecision")


def test_native_schema_changes_after_five_retrievals(provider):
    name, llm, responses, requests = provider
    responses.extend(
        [(json.dumps(decision()), False)] * 5 + [(json.dumps({"answer": "tea"}), False)]
    )
    calls = []
    result = reader_for(llm, calls).invoke(
        {"messages": [HumanMessage(content="question")]}
    )
    assert result["num_tool_calls"] == len(calls) == 5 and result["answer"] == "tea"
    assert result["stop_reason"] == "tool_budget_exhausted"
    assert_schema_request(name, requests[-1], "FinalAnswer")


@pytest.mark.parametrize("bad", ["malformed", "native_call", "empty"])
def test_provider_bad_output_is_blocked_and_repaired(provider, bad):
    _, llm, responses, requests = provider
    payload = (
        "not JSON"
        if bad == "malformed"
        else ""
        if bad == "empty"
        else json.dumps(decision())
    )
    responses.extend(
        [
            (payload, bad == "native_call"),
            (json.dumps(decision("answer", "known")), False),
        ]
    )
    calls = []
    result = reader_for(llm, calls).invoke(
        {"messages": [HumanMessage(content="question")]}
    )
    assert calls == [] and result["num_tool_calls"] == 0
    assert result["answer"] == "known" and len(result["rejections"]) == 1
    assert result["num_model_calls"] == len(requests) == 2


def test_openai_function_transport_counts_only_registered_retrievals():
    integration = pytest.importorskip("langchain_openai")
    import httpx

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        name = body["tools"][0]["function"]["name"]
        payload = {"answer": "tea"} if name == "FinalAnswer" else decision()
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"schema_{len(requests)}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(payload),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    calls = []
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = integration.ChatOpenAI(
            model="gpt-3.5-turbo", api_key="test", http_client=client
        )
        with pytest.warns(UserWarning, match="Overriding to method='function_calling'"):
            reader = reader_for(llm, calls)
        result = reader.invoke({"messages": [HumanMessage(content="question")]})
    assert result["answer"] == "tea" and result["num_tool_calls"] == len(calls) == 5
    assert (
        result["num_model_calls"] == len(requests) == 6 and result["rejections"] == []
    )
    assert requests[-1]["tools"][0]["function"]["name"] == "FinalAnswer"
    assert all(
        len(request["tools"]) == 1 and request["parallel_tool_calls"] is False
        for request in requests
    )

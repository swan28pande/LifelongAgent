# Recall source evidence

This file contains source observations and exact excerpts. Interpretation and confidence are in [conclusions.md](conclusions.md) and [issues.json](issues.json).

## u3_p202711_0137 / u3_q0130

Source category label: `unobserved_shift_boundary`. Reference: `comedy podcast`. Target: 2027-10-24; probe: 2027-11-30.

[datasets/v2/u3/probing_questions.json:4397](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4397)

```text
          "question": "What was Sofia's podcast on Sunday, 2027-10-24?",
```

[datasets/v2/u3/probing_questions.json:4398](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4398)

```text
          "answer": "comedy podcast",
```

[datasets/v2/u3/probing_questions.json:4400](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4400)

```text
          "accept": [
            "comedy podcast"
          ],
```

[datasets/v2/u3/probing_questions.json:4409](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4409)

```text
          "viewpoint_day": 640,
```

[datasets/v2/u3/world_state.json:57604](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/world_state.json:57604)

```text
          "value": "comedy podcast",
```

[datasets/v2/u3/world_state.json:57609](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/world_state.json:57609)

```text
          "mentioned": false,
```

[datasets/v2/u3/world_state.json:2050](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/world_state.json:2050)

```text
      "start": 603,
```

[datasets/v2/u3/world_state.json:2073](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/world_state.json:2073)

```text
      "first_mention_day": 604,
```

[datasets/v2/u3/world_state.json:57422](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/world_state.json:57422)

```text
          "value": "Italian lessons podcast",
```

[datasets/v2/u3/world_state.json:57689](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/world_state.json:57689)

```text
          "value": "dog training podcast",
```

[datasets/v2/u3/sessions/2027-10-22.json:29](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/sessions/2027-10-22.json:29)

```text
      "text": "Actually, yes! I'm going to play an episode of an Italian language lesson podcast. I really want to practice my pronunciation while I commute."
```

[datasets/v2/u3/sessions/2027-10-25.json:29](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/sessions/2027-10-25.json:29)

```text
      "text": "Actually, my podcast routine has completely changed recently. Since we adopted Dash, our rescue greyhound, I've started listening to a show focused on dog training and canine behavior instead."
```

[generator_v2/qa.py:262](/home/cloaked/projects/LifelongAgent/generator_v2/qa.py:262)

```text
            accept = [self.date(d).isoformat() for d in range(reg.start, first + 1)]
            tags = [reg.kind, reg.visibility] + (["bounded"] if first > reg.start else [])
            self.add(Draft("change_detection", reg.domain,
```

## u4_p202605_0012 / u4_q0015

Source category label: `future_target_in_recall`. Reference: `bacon and eggs`. Target: 2026-08-23; probe: 2026-05-31.

[datasets/v2/u4/probing_questions.json:301](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:301)

```text
          "question": "What was Daniel's breakfast on Sunday, 2026-08-23?",
```

[datasets/v2/u4/probing_questions.json:302](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:302)

```text
          "answer": "bacon and eggs",
```

[datasets/v2/u4/probing_questions.json:304](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:304)

```text
          "accept": [
            "bacon and eggs"
          ],
```

[datasets/v2/u4/probing_questions.json:311](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:311)

```text
          "viewpoint_day": 92,
```

[datasets/v2/u4/world_state.json:21116](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/world_state.json:21116)

```text
          "value": "bacon and eggs",
```

[generator_v2/probing.py:90](/home/cloaked/projects/LifelongAgent/generator_v2/probing.py:90)

```text
        if q["type"] == "fact_at_time":
            m = re.search(r"\d{4}-\d{2}-\d{2}", q["question"])
            if m:
                qdate = dt.date.fromisoformat(m.group())
                qday = (qdate - start).days + 1
                effective_day = max(effective_day, qday)
                if effective_day > num_days:
                    continue
        assigned_month = _day_to_month(effective_day, start)
```

## u4_p202707_0104 / u4_q0108

Source category label: `future_target_in_recall`. Reference: `fish and chips`. Target: 2027-08-20; probe: 2027-07-31.

[datasets/v2/u4/probing_questions.json:3269](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3269)

```text
          "question": "What was Daniel's dinner on Friday, 2027-08-20?",
```

[datasets/v2/u4/probing_questions.json:3270](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3270)

```text
          "answer": "fish and chips",
```

[datasets/v2/u4/probing_questions.json:3272](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3272)

```text
          "accept": [
            "fish and chips"
          ],
```

[datasets/v2/u4/probing_questions.json:3280](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3280)

```text
          "viewpoint_day": 518,
```

[datasets/v2/u4/world_state.json:59158](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/world_state.json:59158)

```text
          "value": "fish and chips",
```

[generator_v2/probing.py:90](/home/cloaked/projects/LifelongAgent/generator_v2/probing.py:90)

```text
        if q["type"] == "fact_at_time":
            m = re.search(r"\d{4}-\d{2}-\d{2}", q["question"])
            if m:
                qdate = dt.date.fromisoformat(m.group())
                qday = (qdate - start).days + 1
                effective_day = max(effective_day, qday)
                if effective_day > num_days:
                    continue
        assigned_month = _day_to_month(effective_day, start)
```

## u4_p202710_0132 / u4_q0072

Source category label: `future_target_in_recall`. Reference: `jazz records`. Target: 2027-12-14; probe: 2027-10-31.

[datasets/v2/u4/probing_questions.json:4119](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4119)

```text
          "question": "What was Daniel's work music on Tuesday, 2027-12-14?",
```

[datasets/v2/u4/probing_questions.json:4120](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4120)

```text
          "answer": "jazz records",
```

[datasets/v2/u4/probing_questions.json:4122](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4122)

```text
          "accept": [
            "jazz records"
          ],
```

[datasets/v2/u4/probing_questions.json:4129](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4129)

```text
          "viewpoint_day": 610,
```

[datasets/v2/u4/world_state.json:71497](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/world_state.json:71497)

```text
          "value": "jazz records",
```

[generator_v2/probing.py:90](/home/cloaked/projects/LifelongAgent/generator_v2/probing.py:90)

```text
        if q["type"] == "fact_at_time":
            m = re.search(r"\d{4}-\d{2}-\d{2}", q["question"])
            if m:
                qdate = dt.date.fromisoformat(m.group())
                qday = (qdate - start).days + 1
                effective_day = max(effective_day, qday)
                if effective_day > num_days:
                    continue
        assigned_month = _day_to_month(effective_day, start)
```

## u4_p202801_0153 / u4_q0029

Source category label: `future_target_in_recall`. Reference: `rest day`. Target: 2028-02-04; probe: 2028-01-31.

[datasets/v2/u4/probing_questions.json:5017](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:5017)

```text
          "question": "What was Daniel's workout on Friday, 2028-02-04?",
```

[datasets/v2/u4/probing_questions.json:5018](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:5018)

```text
          "answer": "rest day",
```

[datasets/v2/u4/probing_questions.json:5020](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:5020)

```text
          "accept": [
            "rest day"
          ],
```

[datasets/v2/u4/probing_questions.json:5029](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:5029)

```text
          "viewpoint_day": 702,
```

[datasets/v2/u4/world_state.json:77002](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/world_state.json:77002)

```text
          "value": "rest day",
```

[generator_v2/probing.py:90](/home/cloaked/projects/LifelongAgent/generator_v2/probing.py:90)

```text
        if q["type"] == "fact_at_time":
            m = re.search(r"\d{4}-\d{2}-\d{2}", q["question"])
            if m:
                qdate = dt.date.fromisoformat(m.group())
                qday = (qdate - start).days + 1
                effective_day = max(effective_day, qday)
                if effective_day > num_days:
                    continue
        assigned_month = _day_to_month(effective_day, start)
```

## u5_p202605_0011 / u5_q0006

Source category label: `unobserved_answer_phase`. Reference: `bike ride`. Target: 2026-05-09; probe: 2026-05-31.

[datasets/v2/u5/probing_questions.json:284](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:284)

```text
          "question": "What was Priya's workout on Saturday, 2026-05-09?",
```

[datasets/v2/u5/probing_questions.json:285](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:285)

```text
          "answer": "bike ride",
```

[datasets/v2/u5/probing_questions.json:287](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:287)

```text
          "accept": [
            "bike ride"
          ],
```

[datasets/v2/u5/probing_questions.json:296](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:296)

```text
          "viewpoint_day": 92,
```

[datasets/v2/u5/world_state.json:10734](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:10734)

```text
          "value": "bike ride",
```

[datasets/v2/u5/probing_questions.json:290](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:290)

```text
          "evidence_days": [
            68,
            77,
            79
          ],
```

[datasets/v2/u5/world_state.json:15](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:15)

```text
      "rule": {
        "type": "nested",
        "block_days": 7,
        "blocks": [
          {
            "type": "day_of_week",
            "map": {
              "mon": "evening run",
              "tue": "bouldering",
              "wed": "evening run",
              "thu": "bouldering",
              "fri": "yoga class",
              "sat": "long run",
              "sun": "rest day"
            }
          },
          {
            "type": "day_of_week",
            "map": {
              "mon": "swim",
              "tue": "bouldering",
              "wed": "swim",
              "thu": "bouldering",
              "fri": "yoga class",
              "sat": "bike ride",
              "sun": "rest day"
            }
          }
        ]
      },
```

[datasets/v2/u5/world_state.json:10490](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:10490)

```text
          "value": "bouldering",
```

[datasets/v2/u5/world_state.json:11522](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:11522)

```text
          "value": "long run",
```

[datasets/v2/u5/world_state.json:11753](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:11753)

```text
          "value": "swim",
```

[datasets/v2/u5/sessions/2026-05-07.json:45](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-07.json:45)

```text
      "text": "I actually headed to the gym right after work. Some other folks there were doing heavy weight training, but I spent my session climbing on the short walls with crash pads and no harness."
```

[datasets/v2/u5/sessions/2026-05-16.json:29](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-16.json:29)

```text
      "text": "She is! In fact, her workout today was a grueling open-water swim at the lake. For my own exercise, I decided to do some serious mileage on foot around the local parks after I left the office—I ended up jogging for nearly two hours. I'm definitely feeling it now!"
```

[datasets/v2/u5/sessions/2026-05-18.json:29](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-18.json:29)

```text
      "text": "She's doing incredibly well. She actually did a long run this evening to get her miles in. As for me, I opted for a different kind of exercise after work. I went to do laps in the indoor pool, focusing on my breaststroke and freestyle."
```

[datasets/v2/u5/sessions/2026-05-07.json:13](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-07.json:13)

```text
      "text": "Hey! Yes, I got lucky. My coworker Dana actually took the bus because she was tired from her triathlon training, but I decided to pedal my bicycle to the office today. The fresh air was really nice."
```

[datasets/v2/u5/sessions/2026-05-18.json:13](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-18.json:13)

```text
      "text": "Hey! It's going well. I actually found a nice little seafood bistro near my office that has amazing grilled salmon. Aside from that, I changed up how I got to work today. Instead of my usual routine, I ended up using one of those public bikeshare rentals to pedal to the office. It was nice to be out in the morning air."
```

[datasets/v2/u5/sessions/2026-05-24.json:29](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-24.json:29)

```text
      "text": "No, today was strictly a non-active day. I decided to completely avoid any physical exercise and just give my muscles a full break from bouldering. On the other hand, my coworker Dana, who is still hard at work training for her upcoming triathlon, messaged me that she did a brutal three-hour cycling session in the rain today."
```

[datasets/v2/u5/sessions/2026-05-30.json:13](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-30.json:13)

```text
      "text": "Hi! It's been a really nice, relaxing Saturday so far. I actually didn't have to travel to an office or do any commuting today, which was a fantastic change of pace. Instead, I kicked off my morning by going for a really long jog around the neighborhood. My coworker Dana, who is training for that triathlon, did a cycling session today, but I preferred just hitting the pavement."
```

[generator_v2/qa.py:193](/home/cloaked/projects/LifelongAgent/generator_v2/qa.py:193)

```text
        pool = same or reg.mention_days
        return sorted(sorted(pool, key=lambda d: abs(d - day))[:n])
```

## u5_p202610_0047 / u5_q0048

Source category label: `unobserved_shift_boundary`. Reference: `synthwave playlists`. Target: 2026-09-17; probe: 2026-10-31.

[datasets/v2/u5/probing_questions.json:1317](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1317)

```text
          "question": "What was Priya's work music on Thursday, 2026-09-17?",
```

[datasets/v2/u5/probing_questions.json:1318](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1318)

```text
          "answer": "synthwave playlists",
```

[datasets/v2/u5/probing_questions.json:1320](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1320)

```text
          "accept": [
            "synthwave playlists"
          ],
```

[datasets/v2/u5/probing_questions.json:1329](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1329)

```text
          "viewpoint_day": 245,
```

[datasets/v2/u5/world_state.json:26359](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:26359)

```text
          "value": "synthwave playlists",
```

[datasets/v2/u5/world_state.json:26364](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:26364)

```text
          "mentioned": false,
```

[datasets/v2/u5/world_state.json:1299](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:1299)

```text
      "start": 200,
```

[datasets/v2/u5/world_state.json:1314](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:1314)

```text
      "first_mention_day": 203,
```

[datasets/v2/u5/world_state.json:25735](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:25735)

```text
          "value": "lo-fi beats",
```

[datasets/v2/u5/world_state.json:26604](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:26604)

```text
          "value": "synthwave playlists",
```

[datasets/v2/u5/sessions/2026-09-12.json:37](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-09-12.json:37)

```text
      "text": "I did do a few hours of coding this morning to fix some database issues. To keep my focus, I had some relaxed, wordless instrumental hip-hop tracks playing in the background. Arjun was working on his own project while listening to loud heavy metal, but I definitely needed those chill, repetitive beats to concentrate."
```

[datasets/v2/u5/sessions/2026-09-19.json:29](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-09-19.json:29)

```text
      "text": "I did! I got into a really solid groove. To help me focus, I put on some upbeat instrumental tracks filled with retro-futuristic electronic synthesizer beats. Arjun was working nearby on his own projects, but he was listening to traditional acoustic jazz."
```

[generator_v2/qa.py:262](/home/cloaked/projects/LifelongAgent/generator_v2/qa.py:262)

```text
            accept = [self.date(d).isoformat() for d in range(reg.start, first + 1)]
            tags = [reg.kind, reg.visibility] + (["bounded"] if first > reg.start else [])
            self.add(Draft("change_detection", reg.domain,
```


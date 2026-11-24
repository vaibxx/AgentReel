# AgentReel

AgentReel turns a brief and source documents into a narrated 30–60 second video. Each run saves the evidence, fact IDs, timed script, storyboard, subtitles, review decisions and technical QA results alongside the MP4.

[Watch the 30-second rainbow demo](https://raw.githubusercontent.com/vaibxx/AgentReel/main/demo/rainbow.mp4) · [Read the timed captions](demo/rainbow.srt)

The demo follows one ray through a circular droplet, then shows how light from different droplets reaches the observer. Its script and evidence came from the configured Gemini pipeline. After a provider outage, the final run reused those artifacts, corrected the citations, used the procedural optics storyboard and passed editorial review with Codex CLI. AgentReel rendered the narration, diagrams and captions locally and passed its technical QA checks.

## Reproduce the rainbow example

Install Python 3.11 or a compatible later version and FFmpeg with ffprobe on PATH. On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/setup_models.py
```

Configure the following values privately in a local `.env` file. Use a Gemini model available to your account. Confirm free access only if it applies to your account.

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
LLM_API_KEY=
FREE_API_CONFIRMED=true
```

```powershell
.\.venv\Scripts\python.exe run.py --brief data/briefs/general.yaml --documents data/briefs/rainbow-source.md --seed 42 --stock-source none --image-backend none --video-model none
```

This example uses programmatically drawn visuals. Local voice and embedding weights are downloaded by the setup script. Planning and editorial stages call the configured language model. A seed does not guarantee identical provider responses.

## Inspect the run

Open the run directory printed by the CLI. Read `facts.json` alongside `evidence.json` to see what supports the narration. Compare `storyboard.json` with the video to judge whether the pictures explain those facts. `qa_report.json` checks media properties and timing. A video can pass those checks and still need visual revision.

Run `python -m pytest -q tests` to exercise the test suite. Optional GPU tests depend on additional libraries and cached model fixtures.

## Credits and scope

This general-purpose edition includes original code, one unbranded example and the fonts with their notices. Institutional logos, private uploads, college source collections, credentials, downloaded models and production logs are excluded. The college-specific mode requires additional materials. See LICENSE and THIRD_PARTY.md for attribution.

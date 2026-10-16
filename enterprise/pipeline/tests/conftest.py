import pathlib
import sys
import types

# Make the `pact` package (the harness around the four service files) importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Stub pact.api.models so synthesizer can be imported without the full app.
_api = types.ModuleType("pact.api")
_models = types.ModuleType("pact.api.models")
for _name in ("BriefRevision", "Event", "Job", "SourceUnit", "Workspace"):
    setattr(_models, _name, type(_name, (), {}))
_models.JobStatus = types.SimpleNamespace(done="done", failed="failed", pending="pending")
_models.JobType = types.SimpleNamespace(extract_brief="extract_brief", synthesize_brief="synthesize_brief")
sys.modules.setdefault("pact.api", _api)
sys.modules.setdefault("pact.api.models", _models)

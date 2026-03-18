"""
Interactive Bracket API Server.
Runs Monte Carlo simulations on demand and returns JSON results.
"""

import sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import directly to avoid __init__.py issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "bracket_simulator",
    Path(__file__).parent.parent.parent / "packages" / "simulation" / "bracket_simulator.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
BracketSimulator = mod.BracketSimulator

app = FastAPI(title="March Madness Bracket API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulator instance
simulator: Optional[BracketSimulator] = None


def get_simulator():
    global simulator
    if simulator is None:
        simulator = BracketSimulator()
    return simulator


@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "bracket.html")


@app.post("/api/reset")
async def reset():
    """Reset the bracket to initial state."""
    global simulator
    simulator = BracketSimulator()
    return {"status": "ok", "message": "Bracket reset"}


@app.post("/api/simulate/first-four")
async def simulate_first_four(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate First Four games."""
    sim = get_simulator()
    results = sim.simulate_round("first_four", n_sims)
    return results


@app.post("/api/simulate/r64")
async def simulate_r64(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate Round of 64."""
    sim = get_simulator()
    results = sim.simulate_round("r64", n_sims)
    return results


@app.post("/api/simulate/r32")
async def simulate_r32(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate Round of 32."""
    sim = get_simulator()
    results = sim.simulate_round("r32", n_sims)
    return results


@app.post("/api/simulate/s16")
async def simulate_s16(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate Sweet 16."""
    sim = get_simulator()
    results = sim.simulate_round("s16", n_sims)
    return results


@app.post("/api/simulate/e8")
async def simulate_e8(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate Elite 8."""
    sim = get_simulator()
    results = sim.simulate_round("e8", n_sims)
    return results


@app.post("/api/simulate/f4")
async def simulate_f4(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate Final Four."""
    sim = get_simulator()
    results = sim.simulate_round("f4", n_sims)
    return results


@app.post("/api/simulate/championship")
async def simulate_championship(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate Championship."""
    sim = get_simulator()
    results = sim.simulate_round("championship", n_sims)
    return results


@app.post("/api/simulate/all")
async def simulate_all_remaining(n_sims: int = Query(default=100_000, ge=1000, le=1_000_000)):
    """Simulate all remaining rounds."""
    sim = get_simulator()
    results = sim.simulate_all_remaining(n_sims)
    return results


@app.get("/api/state")
async def get_state():
    """Get current bracket state."""
    sim = get_simulator()
    return sim.get_state()


@app.get("/api/advancement")
async def get_advancement(n_sims: int = Query(default=50_000, ge=1000, le=500_000)):
    """Get forward Monte Carlo advancement probabilities."""
    sim = get_simulator()
    return sim.calculate_advancement(n_sims)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2502)

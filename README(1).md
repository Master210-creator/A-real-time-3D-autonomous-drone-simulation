# 🚁 Autonomous Drone Simulation v2.0

A real-time 3D autonomous drone simulation built with Python, Pygame, and OpenGL. The drone navigates procedurally generated environments, avoids obstacles, and completes mission objectives — all with a live HUD, memory mapping, and dynamic lighting.

---

## Features

- **Three mission modes** selectable from the main menu
- **Frontier-based exploration** with a live memory/coverage map
- **Realistic 3D environments** — city buildings with windows, forest trees with sway animation, fallen logs
- **Dynamic spotlight lighting** — the drone illuminates only what's beneath it (night modes)
- **Firefly particle system** in forest mode
- **Delivery missions** with address-based pickup/dropoff selection
- **Custom search-area drawing** for obstacle avoidance missions
- **Live HUD** showing state, battery, coverage %, speed, heading, and a mini-map

---

## Mission Modes

| Key | Mode | Description |
|-----|------|-------------|
| `1` | Night Rescue — Urban | Search for survivors in a dark city; drone spotlight is your only visibility |
| `2` | Night Rescue — Forest | Dense forest with fireflies; trees sway, survivors are hidden |
| `3` | Daylight Delivery | Select pickup/dropoff addresses; drone navigates between them |

> **Note:** A fourth mode (Obstacle Avoidance with custom area drawing) is accessible internally via map type 3 in code.

---

## Requirements

- Python 3.9 or higher
- A system with OpenGL support (any modern GPU)

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pygame numpy PyOpenGL PyOpenGL_accelerate
```

> On Linux (Debian/Ubuntu), you may need to add `--break-system-packages` if using the system Python.

---

## Running the Simulation

```bash
python drone_sim.py
```

---

## Controls

| Key / Input | Action |
|-------------|--------|
| `1` / `2` / `3` | Select mission from menu |
| `SPACE` | Pause / Resume |
| `R` | Reset mission |
| `C` | Cycle camera mode (Follow → Top-Down → Free Orbit) |
| `ESC` | Quit |
| **Mouse drag** | Rotate camera (in Follow / Free Orbit modes) |
| **Scroll wheel** | Zoom in / out |

### Delivery Mode (Mode 3) Setup Screen
| Key | Action |
|-----|--------|
| `UP` / `DOWN` | Change selected address |
| `TAB` | Switch between Pickup and Destination fields |
| `ENTER` | Confirm and launch |
| `ESC` | Cancel and return to menu |

---

## Project Structure

```
drone_sim.py          # Main simulation file (single-file project)
requirements.txt      # Python dependencies
README.md             # This file
```

---

## How the AI Works

The drone uses a **finite state machine** with five states:

| State | Behaviour |
|-------|-----------|
| `EXPLORE` | Moves to nearest frontier cell on the memory map |
| `APPROACH` | Heads toward a detected target |
| `RESCUE` | Marks the target as found / picks up parcel |
| `RETURN` | Returns to origin when battery drops below 15% |
| `AVOID` | Backs away and randomizes heading when stuck |

**Memory map** cells are illuminated as the drone flies over them; edge detection highlights the boundary of explored vs. unexplored territory on the mini-map.

---

## Known Limitations

- Single-drone simulation (no multi-agent)
- Pathfinding is frontier-based greedy, not full A* — the drone may take suboptimal routes in dense environments
- PyOpenGL_accelerate is optional but recommended for better frame rates; the sim will auto-install it if missing

---

## License

MIT — free to use, modify, and distribute.

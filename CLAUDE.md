# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Melting Pot is a multi-agent reinforcement learning research framework designed to evaluate agent generalization to novel social situations. Built on DeepMind Lab2D, it provides over 50 multi-agent game environments (substrates) and 256+ test scenarios for assessing cooperation, competition, deception, reciprocation, trust, and other social behaviors.

### Research Purpose
- Tests agent performance across diverse social interactions with both familiar and unfamiliar individuals
- Quantifies generalization ability to novel social situations where agents are interdependent
- Serves as a benchmark for ranking multi-agent RL algorithms by social generalization capability

### This Project's Research Focus

This repository extends Melting Pot to explore **novel agent forms and moral conceptual possibilities beyond human limitations**. The research investigates how agents with different self-understanding structures (individual, collective, multiple-agents-within-individual) might develop novel social concepts and practices in multi-agent environments.

**Core Thesis**: The "machine perspective" on moral and social concepts is not constrained by human limitations (cognitive, physical, psychological, social, temporal). By experimenting with agents that transcend these boundaries, new moral and social possibilities can emerge that illustrate conceptual spaces beyond what humans can realize.

**Experimental Approach**:
- Design agents with novel identity structures and self-understanding
- Observe emergent social behaviors and concepts in Melting Pot environments
- Analyze how different agent architectures lead to new forms of cooperation, conflict resolution, and moral reasoning
- Use findings to demonstrate expanded possibilities for moral and social concepts

### Technical Foundation
- **Engine**: DeepMind Lab2D (2D grid-based game engine with Lua scripting)
- **Grid System**: 3D coordinates (x, y, layer) where only one piece occupies each position
- **Piece-based**: Game entities move on grid like chess pieces, with 4 cardinal orientations
- **Event-driven**: Logic implemented via callbacks for state changes, collisions, and interactions

## Common Development Commands

### Installation and Setup
- Install dependencies: `pip install --editable .[dev]` or use `./bin/install.sh`
- The project uses a virtual environment: `venv_meltingpot/`

### Testing
- Run all tests: `pytest meltingpot` or `pytest --pyargs meltingpot`
- Run example tests: `pytest examples`
- Run specific test: `pytest meltingpot/path/to/specific_test.py`

### Code Quality
- Lint code: `pylint --errors-only meltingpot`
- Type checking: `pytype meltingpot`
- Format code: `pyink` (configured in pyproject.toml)
- Sort imports: `isort` (configured for Google style)

### Interactive Play
- Play substrates interactively: `python meltingpot/human_players/play_<substrate_name>.py`
- Example: `python meltingpot/human_players/play_clean_up.py`
- Controls: WASD for movement, Q/E for turning, 1/2 for actions, TAB to switch players

## Architecture

### Core Components
- **Substrates** (`meltingpot/substrate.py`): Multi-agent game environments
- **Scenarios** (`meltingpot/scenario.py`): Specific test configurations with predefined bot populations
- **Bots** (`meltingpot/bot.py`): Agent policies including saved models and puppeteers

### Key Directories
- `meltingpot/configs/`: Configuration files for substrates, scenarios, and bots
- `meltingpot/lua/`: Lua level definitions and game logic (built on Lab2D)
- `meltingpot/utils/`: Core utilities including policies, evaluation, and substrate builders
- `meltingpot/human_players/`: Interactive scripts for manual gameplay
- `examples/`: Training examples (RLlib) and tutorials
- `agency_experiments/`: Research experiments directory
- `neural_training/`: Neural network training scripts

### Framework Structure
- Built on **DeepMind Lab2D** engine with Lua scripting for game logic
- Uses **GameObject/Component** architecture in Lua for game entities
- Python wrapper provides RL environment interface (dm_env compatible)
- Supports TensorFlow SavedModel policies and custom puppeteers

### GameObject/Component System
- **GameObject**: Empty container that holds Components (avatars, walls, items, spawn points)
- **Component**: Modular game logic with lifecycle callbacks (`awake`, `reset`, `start`, `update`, etc.)
- **StateManager**: Required component managing object states, layers, sprites, groups, and contacts
- **Transform**: Required component handling position, orientation, movement, and spatial queries
- **Appearance**: Manages visual rendering, sprites, and palettes
- **Event System**: Components respond to `onEnter`, `onExit`, `onHit`, `onBlocked`, `onStateChange`

### Engine Update Cycle
1. Render all objects (finalizes player observations)
2. Call component `update()` functions in arbitrary order
3. Run registered updaters in priority order with fine-grained control
4. Process queued events (movement, state changes, collisions, beam hits)

### Configuration System
- Substrates defined in `meltingpot/configs/substrates/`
- Scenarios combine substrates with bot populations
- Uses ml_collections.ConfigDict for configuration management

### Assets
- Game assets downloaded automatically during installation from Google Cloud Storage
- Stored in `meltingpot/assets/` (auto-populated)
- Includes pre-trained bot models in `assets/saved_models/`

## Creating New Substrates

### Substrate Structure
A substrate requires:
1. **Lua files** in `meltingpot/lua/levels/<substrate_name>/`:
   - `init.lua`: API Factory with Simulation settings
   - `components.lua`: Custom components (optional)
2. **Python config** in `meltingpot/configs/substrates/<substrate_name>.py`
3. **Human player script** in `meltingpot/human_players/play_<substrate_name>.py`

### Key Concepts
- **Prefabs**: Templates for GameObjects defined in Python config
- **GameObject**: Entity containers that hold Components
- **Components**: Modular game logic (StateManager, Transform, Appearance, etc.)
- **ASCII Maps**: Define spatial layout using character-to-prefab mapping

### Required Components
- `StateManager`: Manages object states, layers, and sprites
- `Transform`: Handles position and orientation
- `Appearance`: Controls visual rendering and sprites

### Development Workflow
1. Create Lua level files in `meltingpot/lua/levels/<name>/`
2. Define prefabs and ASCII map in Python config
3. Register custom components in Lua component registry
4. Create human player script for interactive testing
5. Add substrate to `meltingpot/configs/substrates/__init__.py`

### Testing New Substrates
- Interactive testing: `python meltingpot/human_players/play_<substrate>.py`
- Debug mode: Add `--verbose True` flag to see component interactions
- Run substrate tests: `pytest meltingpot/configs/substrates/<substrate>_test.py`
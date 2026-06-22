import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


def test_leader_name_survives_agent_insertion():
    dummy = main.AgentSpec(
        name="dummy_root",
        initial_state=main.AGENTS[0].initial_state.copy(),
        destination=main.AGENTS[0].destination.copy(),
    )
    specs = main.resolve_leader_references([dummy, *main.AGENTS])

    right_turn_nb = next(i for i, agent in enumerate(specs) if agent.name == "right_turn_northbound")
    follower_lt = next(i for i, agent in enumerate(specs) if agent.name == "follower_left_turn")

    assert specs[right_turn_nb].leader_index == follower_lt


def test_missing_leader_name_fails_clearly():
    specs = [replace(agent) for agent in main.AGENTS]
    right_turn_nb = next(i for i, agent in enumerate(specs) if agent.name == "right_turn_northbound")
    specs[right_turn_nb] = replace(specs[right_turn_nb], leader_name="not_a_vehicle")

    with pytest.raises(ValueError, match="missing leader_name"):
        main.resolve_leader_references(specs)


def test_leader_cycle_fails_clearly():
    specs = [replace(agent) for agent in main.AGENTS]
    nb_lt = next(i for i, agent in enumerate(specs) if agent.name == "left_turn_northbound")
    follower_lt = next(i for i, agent in enumerate(specs) if agent.name == "follower_left_turn")
    specs[nb_lt] = replace(specs[nb_lt], leader_name="follower_left_turn")
    specs[follower_lt] = replace(specs[follower_lt], leader_name="left_turn_northbound")

    with pytest.raises(ValueError, match="leader cycle"):
        main.resolve_leader_references(specs)

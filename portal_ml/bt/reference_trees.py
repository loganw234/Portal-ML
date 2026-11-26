"""Reference Behavior Tree in DSL for unit testing and simulator validation.

Note: The evolutionary trainer does NOT seed the population with this tree;
it evolves solutions from random ASTs. This tree exists only as an oracle to
verify that the simulator and Source bridge are solvable and physically sound.
"""

from __future__ import annotations

from portal_ml.bt.ast import BTNode, parse_bt

REFERENCE_SOLVER_DSL = """
SEQUENCE(
  SELECTOR(
    SEQUENCE(
      CONDITION(is_open(exit)),
      ACTION(SetTarget(exit))
    ),
    SEQUENCE(
      CONDITION(exists(pedestal_button)),
      ACTION(SetTarget(pedestal_button))
    ),
    SEQUENCE(
      CONDITION(exists(cube)),
      CONDITION(NOT is_holding(cube)),
      ACTION(SetTarget(cube))
    ),
    SEQUENCE(
      CONDITION(exists(floor_button)),
      ACTION(SetTarget(floor_button))
    )
  ),
  SELECTOR(
    SEQUENCE(
      CONDITION(NOT is_reachable(target)),
      SELECTOR(
        SEQUENCE(
          CONDITION(portals_bridge(target)),
          ACTION(MoveTo(portal_here))
        ),
        SEQUENCE(
          CONDITION(NOT portal_in_zone(blue, here)),
          ACTION(ShootPortal(blue, wall_here))
        ),
        ACTION(ShootPortal(orange, wall_near_target))
      )
    ),
    SEQUENCE(
      CONDITION(in_range(target)),
      SELECTOR(
        ACTION(Press(target)),
        ACTION(Grab(target)),
        SEQUENCE(
          CONDITION(target_is(floor_button)),
          ACTION(Drop())
        )
      )
    ),
    ACTION(MoveTo(target))
  )
)
""".strip()


def get_reference_tree() -> BTNode:
    return parse_bt(REFERENCE_SOLVER_DSL)

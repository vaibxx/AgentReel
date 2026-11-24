import math

import pytest

from agentreel.rainbow import SPECTRUM, path_for
from agentreel.rainbow import RainbowDiagram


@pytest.mark.parametrize("color,index", SPECTRUM)
def test_primary_ray_contacts_the_surface_three_times(color, index):
    source, entry, reflection, exit_point, outgoing = path_for(index)
    for point in (entry, reflection, exit_point):
        assert math.hypot(*point) == pytest.approx(1)
    assert source[1] == entry[1]
    assert outgoing[0] < exit_point[0]
    assert outgoing[1] > exit_point[1]


def test_red_and_violet_emerge_at_different_angles():
    def angle(index):
        exit_point, end = path_for(index)[-2:]
        return math.atan2(end[1]-exit_point[1], end[0]-exit_point[0])
    assert 0.01 < abs(angle(SPECTRUM[0][1])-angle(SPECTRUM[-1][1])) < 0.1


@pytest.mark.parametrize("narration,phase", [
    ("A rainbow forms in the sky.", "arc"),
    ("White light enters the droplet.", "entry"),
    ("Some light reflects inside the drop.", "reflection"),
    ("It bends on its way out.", "exit"),
    ("Light from many drops reaches your eyes.", "observer"),
])
def test_diagram_phase_draws_visible_motion(narration, phase):
    assert RainbowDiagram.phase(narration) == phase
    diagram = RainbowDiagram((270,480), narration, 3)
    assert diagram.frame(0).tobytes() != diagram.frame(2).tobytes()

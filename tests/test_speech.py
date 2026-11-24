import pytest
from types import SimpleNamespace
from agentreel.models import Brief
from agentreel.speech import check_length,NarrationLengthError

def script(words_per_beat,beats=6):
    return SimpleNamespace(beats=[SimpleNamespace(narration=' '.join(['word']*words_per_beat)) for _ in range(beats)])

def test_overlong_narration_gives_measured_word_target():
    brief=Brief(topic='physics',duration_s=40)
    with pytest.raises(NarrationLengthError) as error: check_length(brief,script(15),[8.1]*6)
    # 90 words took 48.6 s; ~33.3 s of speech fits, so roughly 62 words.
    assert error.value.target_words==62 and 'approximately 62 words' in str(error.value)

def test_fitting_narration_passes():
    brief=Brief(topic='physics',duration_s=40)
    tempo,gap=check_length(brief,script(12),[5.4]*6)
    assert .9<=tempo<=1.12 and .12<=gap<=1.5

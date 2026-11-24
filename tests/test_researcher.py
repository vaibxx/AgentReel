from agents.researcher import verbatim

TEXT='One such startup is a student-made, India’s first UPI enabled charger. The startup, Ayka, is made by seven students. It is incubated at riidl.'

def test_exact_and_typographic_variants_pass():
    assert verbatim("India's first UPI enabled charger",TEXT)
    assert verbatim('The startup, Ayka, is made by seven students.',TEXT)

def test_ellipsis_excerpts_must_all_appear_in_order():
    assert verbatim('One such startup is a student-made... is made by seven students... incubated at riidl',TEXT)
    assert not verbatim('incubated at riidl... One such startup is a student-made',TEXT)

def test_invented_text_fails():
    assert not verbatim('Eye of Ender Blaze Powder + Ender Pearl',TEXT)
    assert not verbatim('...',TEXT)

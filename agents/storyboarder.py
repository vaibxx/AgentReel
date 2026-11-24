from agentreel.models import Storyboard

def plain_animation(board):
    """Animation shots that explain no formula: candidates for a detailed illustration."""
    return [s.beat_index for s in board.shots if s.visual_mode=='animation' and not any(o.shape=='text' and any(c in o.label for c in '=→+⇌') for o in s.scene.objects)]

def run(brief,script,provider,ctx,illustrations=False,plentiful=False):
    if brief.visual_template == "rainbow_optics":
        from agentreel.rainbow import RainbowDiagram
        from agentreel.models import Shot
        descriptions = {
            "arc": "Five colored arcs sweep across the sky facing away from the sun.",
            "entry": "A horizontal white ray reaches a stationary circular droplet. Colored paths bend at its rim according to Snell's law.",
            "reflection": "At the same scale, the rays progress to the rear rim and reflect inward. The exit point remains on the same circle.",
            "exit": "The same paths reflect inside the droplet, then bend on exit and diverge toward the observer's side.",
            "observer": "Side view: a ray travels from the Sun on the left toward rain on the right; returning light reaches an eye between them. The observer faces the rain.",
        }
        shots = [Shot(beat_index=i, visual="path", label=beat.headline[:32],
                      accent="teal", visual_mode="animation", demonstration="optics",
                      visual_description=descriptions[RainbowDiagram.phase(beat.narration)])
                 for i, beat in enumerate(script.beats)]
        board = Storyboard(shots=shots)
        ctx.artifact("storyboard.json", board.model_dump(), "storyboarder")
        return board
    feedback=''
    wants_stock=brief.mode=='general' and brief.brand!='somaiya'
    for attempt in range(2):
        board=provider.generate('storyboarder',Storyboard,{'brief':brief.model_dump(),'script':script.model_dump(),'stock_footage_available':wants_stock,'illustration_available':illustrations,'illustrations_plentiful':bool(illustrations and plentiful)},feedback)
        if sorted(s.beat_index for s in board.shots)==list(range(len(script.beats))) and all(s.scene is not None for s in board.shots):
            missing=[s.beat_index for s in board.shots if not s.stock_query.strip()]
            # Footage-led films need a query on most shots; one repair attempt, then accept.
            if wants_stock and attempt==0 and len(missing)*2>len(board.shots):
                feedback=f'Shots {missing} have an empty stock_query. Give EVERY shot a concrete 2-3 word filmable stock_query and 2-3 broader stock_alternates.'
                continue
            # A film of plain diagrams wastes the detailed illustrations this machine can paint.
            plain=plain_animation(board)
            wanted=2 if plentiful else 1
            if illustrations and attempt==0 and len(plain)>=3 and sum(s.visual_mode=='illustration' for s in board.shots)<wanted:
                feedback=(f'illustration_available is true, but shots {plain} are animation without any formula. Convert the 2-3 of them that would look best '
                          'as a detailed painted image to visual_mode=illustration with a vivid image_prompt (subject first, setting, lighting, mood; no real people, '
                          'no trademarked characters or logos). Keep animation only for formulas, labelled processes and comparisons.')
                continue
            board.shots.sort(key=lambda s:s.beat_index)
            if not illustrations:
                # Without a GPU, illustration shots render their animated scene instead.
                for s in board.shots:
                    if s.visual_mode=='illustration': s.visual_mode='animation'
            ctx.artifact('storyboard.json',board.model_dump(),'storyboarder'); return board
        feedback='Return exactly one shot per beat, with unique consecutive zero-based beat_index values. EVERY shot requires an executable scene with subject objects, motion poses and a visible explanatory change; generic legacy templates are not accepted.'
    raise ValueError('Storyboard must cover every beat with an executable scene. No unrelated-template fallback is allowed.')

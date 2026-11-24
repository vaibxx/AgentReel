from agentreel.models import EditorialReview
def editorial(brief,facts,script,provider,ctx,board=None):
    data={'brief':brief.model_dump(),'facts':facts,'script':script.model_dump(),'storyboard':board.model_dump() if board else None,'check':'Every factual claim including visual labels and contacts, required inclusions, forbidden content, first-three-second hook, CTA, institutional scope, dates, safety.'}
    if brief.visual_template == "rainbow_optics":
        from agentreel.rainbow import RainbowDiagram, SPECTRUM, path_for, sky_geometry
        data["executed_diagram"] = {
            "coordinates": "Unit circle, positive y downward. Each path is source, entry, internal reflection, exit, emerging ray endpoint.",
            "paths": {color: path_for(n) for color,n in SPECTRUM},
            "phases": [RainbowDiagram.phase(b.narration) for b in script.beats],
            "sky_geometry_pixels": sky_geometry(brief.size),
            "sky_projection": "Arc phase draws each listed radius from 180 through 360 degrees. In observer phase each drop is positioned on its color's actual exit sightline extrapolated backward from the common eye. Parallel input beams arrive from sun x at that drop's y. The five return rays end at the eye. Sun, rain and observer are labeled. During the last third the side elevation dissolves into the eye's view of the five arcs in corresponding spectrum order.",
            "renderer": "Procedural diagram draws the shared geometry directly. Layout and progress for each phase are described by the shot. There are no generic scene-graph objects in this mode."
        }
    review=provider.generate('critic',EditorialReview,data)
    ctx.artifact(f'editorial_review_{ctx.calls:03}.json',review.model_dump(),'critic')
    return review

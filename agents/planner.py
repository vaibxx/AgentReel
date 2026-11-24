from agentreel.models import Plan
def run(brief,provider,ctx):
    plan=provider.generate('planner',Plan,{'brief':brief.model_dump(),'available_time_s':10800,'api_budget_inr':0})
    ctx.artifact('plan.json',{**plan.model_dump(),'stage_time_budget_s':{'researcher':900,'scriptwriter':600,'storyboarder':300,'asset_generator':3600,'editor':3600,'critic':1200},'api_budget_inr':0},'planner')
    return plan

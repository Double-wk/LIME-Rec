"""Opt-in schema-constrained local-controller variants; never reuse old responses."""
import json

import requests

from lime_rec.agentic.budgeted import LLMSelector
from lime_rec.agentic.llm_client import OpenAICompatibleClient


def action_schema(payload):
    available=[t for t in payload['available_tools'] if t not in payload['called_tools']]
    if 'candidate_ids' not in payload:
        return {'type':'object','properties':{'tool':{'type':'string','enum':available}},
                'required':['tool'],'additionalProperties':False}
    finish={'type':'object','properties':{'action':{'const':'finish'},'ranking':{
        'type':'array','items':{'type':'string','enum':payload['candidate_ids']},'minItems':10,
        'maxItems':len(payload['candidate_ids']),'uniqueItems':True}},
        'required':['action','ranking'],'additionalProperties':False}
    if not available or payload['remaining_tool_budget']==0:return finish
    call={'type':'object','properties':{'action':{'const':'call_tool'},
        'tool':{'type':'string','enum':available}},'required':['action','tool'],'additionalProperties':False}
    return {'anyOf':[call,finish]}


class SchemaClient(OpenAICompatibleClient):
    def generate(self,messages,*,temperature,max_tokens):
        content=messages[-1]['content']
        payload=json.loads(content if content.lstrip().startswith('{') else content.split('\n',1)[1])
        response=requests.post(self.base_url.rstrip('/')+'/chat/completions',
            headers={'Authorization':f'Bearer {self.api_key}','Content-Type':'application/json'},
            json={'model':self.model_name,'messages':list(messages),'temperature':temperature,'max_tokens':max_tokens,
                  'response_format':{'type':'json_schema','json_schema':{'name':'audited_action',
                        'strict':True,'schema':action_schema(payload)}}},timeout=self.timeout)
        response.raise_for_status()
        value=response.json()
        return {'text':value['choices'][0]['message']['content'],'usage':value.get('usage',{}),'raw':value}


class RetryingSelector(LLMSelector):
    def __init__(self,client,max_tokens=128,retries=2):
        super().__init__(client,max_tokens)
        if retries<0:raise ValueError('nonnegative retry count required')
        self.retries=retries

    def __call__(self,view):
        for attempt in range(self.retries+1):
            try:
                tool=super().__call__(view)
                if tool not in view['available_tools']:raise ValueError('unavailable tool')
                return tool
            except ValueError:
                if attempt==self.retries:raise

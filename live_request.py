import json
import requests

def get_comp_name(id: str) -> str:
    url = f'https://worldcubeassociation.org/api/v0/competitions/{id}'
    response = requests.get(url).json()
    return response['name']


def fetch_graphql(url: str, operation_name: str, variables: str, query: str):
    headers = {'Content-Type': 'application/json'}
    data = {'operationName': operation_name, 'variables': variables, 'query': query}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


def get_live_comp_id(id: str) -> int:

    name = get_comp_name(id)

    query_competitions = '''
    query Competitions($filter: String!) {
        competitions(filter: $filter, limit: 10) {
            id
            wca_id
        }
    }
    '''
    url = 'https://live.worldcubeassociation.org/api'
    data = {'filter': name}
    response = fetch_graphql(url, 'Competitions', data, query_competitions)

    competition = next((comp for comp in response['data']['competitions'] if comp['wca_id'] == id), None)
    if not competition:
        return None
    
    return competition['id']


def get_round_id(live_id: int, event_name: str, round_num: str):

    queryRounds = """query Competition($id: ID!) {
          competition(id: $id) {
              competitionEvents {
                  event {
                      id
                  }
                  rounds {
                      id
                      number
                  }
              }
          }
      }"""

    roundData = fetch_graphql(
        'https://live.worldcubeassociation.org/api',
        'Competition',
        { "id": live_id },
        queryRounds
    )

    rounds = roundData['data']['competition']['competitionEvents']
    
    for event in rounds:
        if event['event']['id'] == event_name:
            for round in event['rounds']:
                if round['number'] == round_num:
                    return round['id']
                
    return None


def get_results(round_id: int):
    query = """
    query Round($id: ID!) {
        round(id: $id) {
            id
            name
            competitionEvent {
                id
                event {
                    id
                }
            }
            results {
                id
                ...roundResult
            }
        }
    }

    fragment roundResult on Result {
        attempts {
            result
        }
        person {
            registrant_id
            name
        }
    }
    """
    url = 'https://live.worldcubeassociation.org/api'
    data = {'id': round_id}
    return fetch_graphql(url, 'Round', data, query)['data']['round']['results']

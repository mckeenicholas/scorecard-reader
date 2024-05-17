import sys
import glob
import os
import cv2

import live_request
import warp

EVENTS = {'333': 5, '222': 5, '444': 5, '555': 5, '666': 3, '777': 3, 'sq1': 5, 'skewb': 5, 'pyram': 5, 'minx' :5, 'clock': 5, '333oh': 5, '333bf': 3, '444bf': 3, '555bf': 3}

dir = "raw_images"
dest = "labeled_images"

def label_scorecards(results_formatted):
    for image in glob.glob(f"{dir}{os.sep}*.jpg"):
        filename = os.path.splitext(os.path.basename(image))[0]
        scorecard_id = int(filename)
        res = warp.process_image(image)
        for idx, (live, img) in enumerate(zip(results_formatted[scorecard_id], res)):
            name = live if live > 0 else "DNF"
            cv2.imwrite(f"{dest}{os.sep}{scorecard_id}A{idx}_{name}.jpg", img)

def format_results(results):
    ret = {}
    for result in results:
        attempts = [attempt['result'] for attempt in result['attempts']]
        ret[result['person']['registrant_id']] = attempts
    
    return ret


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python get_images.py <competition_name> <event_name> <round_num>")
        exit(1)
    competition_name = sys.argv[1]
    event_name = sys.argv[2]
    round_num = int(sys.argv[3])

    if event_name not in EVENTS:
        print(f"Event \"{event_name}\" is not supported")
        exit(1)

    if not 1 <= round_num <= 4:
        print(f"Round number {round_num} must be in range [1, 4]")
        exit(1)

    live_comp_id = live_request.get_live_comp_id(competition_name)

    round_id = live_request.get_round_id(live_comp_id, event_name, round_num)

    results = live_request.get_results(round_id)

    results_formatted = format_results(results)

    label_scorecards(results_formatted)

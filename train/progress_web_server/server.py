import os
import re
from bottle import route, run, static_file, template
import pdb
import meta_evaluation

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ROOT = os.path.join(SCRIPT_DIR, '..', '..', 'experiments')
STATUS_TYPES = ['NOT_STARTED', 'RUNNING', 'DONE', 'ERROR', 'KILLED']

def get_status(exp_name):
    for status_type in STATUS_TYPES:
        if os.path.exists(os.path.join(EXP_ROOT, exp_name, status_type)):
            return status_type
    return 'N/A'

def extract_notes(exp_name):
    exp_full_path = os.path.join(EXP_ROOT, exp_name)
    with open(os.path.join(exp_full_path, 'README.md'), 'r') as f:
        readme_tag = f.read()
    m = re.search('Other notes:\s((.*\s*)*)', readme_tag)
    return m.group(1)

@route('/')
def root():
    ret = '<title>Home</title>'
    ret += '<link rel="stylesheet" href="/css" />'

    # Start leaderboard section
    ret += '<div id="leaderboard"><h1>Leaderboard</h1>'
    # Go through each evaluation metric and find best models for the metric
    for perf_info in meta_evaluation.display_info:
        ret += '<h2>%s</h2>' % perf_info[0]
        ret += '<p>'
        # Get sorted models
        model_values_map = meta_evaluation.get_model_values_map()
        overall_perf_tuples = meta_evaluation.sort_exps_by_overall_perf(model_values_map, perf_info)
        for exp_num, iter_num, best_overall_perf in overall_perf_tuples[:5]:
            ret += '\t%f (experiment %d, iter %d)<br>' % (best_overall_perf, exp_num, iter_num)
    # End leaderboard section
    ret += '</p></div>'

    # Start experiments table
    ret += '<div id="experiments"><h1>Experiments</h1><table>'
    # Write table header
    ret += '<tr><th>Experiment</th><th>Status</th><th>Notes</th></tr>'
    for exp_name in sorted(os.listdir(EXP_ROOT)):
        # Experiment tag, with link to details
        exp_tag = '<td><a href="/progress/%s">%s</a></td>' % (exp_name, exp_name)
        # Status tag
        status = get_status(exp_name)
        if status == 'RUNNING':
            status_tag = '<td class="running">%s</td>' % status
        else:
            status_tag = '<td>%s</td>' % status
        # Other notes tag
        other_notes_tag = '<td>%s</td>' % extract_notes(exp_name).replace(os.linesep, '<br>')
        # Write row
        ret += '<tr>' + exp_tag + status_tag + other_notes_tag + '</tr>'
    # End table
    ret += '</table></div>'
    return ret

@route('/progress/<exp_name>')
def progress(exp_name):
    title_tag = '<title>' + exp_name + '</title>'
    css_tag = '<link rel="stylesheet" href="/css" />'

    # Add readme contents
    exp_full_path = os.path.join(EXP_ROOT, exp_name)
    with open(os.path.join(exp_full_path, 'README.md'), 'r') as f:
        readme_contents = f.read()
    readme_contents = readme_contents.replace(os.linesep, '<br>')
    readme_tag = '<h1>Summary</h1>' + readme_contents

    # Add progress plot, if it exists
    if os.path.exists(os.path.join(exp_full_path, 'progress', 'plots.png')):
        image_tag = '<h1>Training plot</h1><img width="100%%" src="/progress/plot/%s" />' % exp_name
    else:
        image_tag = ''

    # Add evaluation results
    evaluation_contents = ''
    model_values_map = meta_evaluation.get_model_values_map(only_include_experiments=[exp_name])
    if model_values_map:
        evaluation_contents = '<h1>Evaluation results</h1>'
        for perf_info in meta_evaluation.display_info:
            evaluation_contents += '<h2>Models sorted by: %s</h2>' % perf_info[0]
            overall_perf_tuples = meta_evaluation.sort_models_by_indiv_perf(model_values_map, perf_info)
            for exp_num, iter_num, best_overall_perf in overall_perf_tuples[:10]:
                evaluation_contents += '%f (iter %d)<br>' % (best_overall_perf, iter_num)
    
    return title_tag + css_tag  + image_tag + readme_tag + evaluation_contents

@route('/progress/plot/<exp_name>')
def plot(exp_name):
    return static_file('plots.png', root=os.path.join(EXP_ROOT, exp_name, 'progress'))

@route('/css')
def css():
    return static_file('style.css', root=SCRIPT_DIR)

run(port=8080, debug=True)

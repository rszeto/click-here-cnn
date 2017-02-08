import os
import re
from bottle import route, run, static_file, template
import pdb

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ROOT = os.path.join(SCRIPT_DIR, '..', '..', 'experiments')
STATUS_TYPES = ['NOT_STARTED', 'RUNNING', 'ERROR', 'KILLED']

def get_status(exp_name):
    for status_type in STATUS_TYPES:
        if os.path.exists(os.path.join(EXP_ROOT, exp_name, status_type)):
            return status_type
    return 'N/A'

def extract_notes(exp_name):
    exp_full_path = os.path.join(EXP_ROOT, exp_name)
    with open(os.path.join(exp_full_path, 'README.md'), 'r') as f:
        readme_contents = f.read()
    m = re.search('Other notes:\s((.*\s*)*)', readme_contents)
    return m.group(1)

@route('/')
def root():
    ret = '<title>Home</title>'
    ret += '<link rel="stylesheet" href="/css" />'
    ret += '<table>'
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
    ret += '</table>'
    return ret

@route('/progress/<exp_name>')
def progress(exp_name):
    title_tag = '<title>' + exp_name + '</title>'
    css_tag = '<link rel="stylesheet" href="/css" />'

    exp_full_path = os.path.join(EXP_ROOT, exp_name)
    with open(os.path.join(exp_full_path, 'README.md'), 'r') as f:
        readme_contents = f.read()
    readme_contents = readme_contents.replace(os.linesep, '<br>')

    # Add progress plot, if it exists
    if os.path.exists(os.path.join(exp_full_path, 'progress', 'plots.png')):
        image_tag = '<img width="100%%" src="/progress/plot/%s" />' % exp_name
    else:
        image_tag = ''

    return title_tag + css_tag + readme_contents + '<br>' + image_tag

@route('/progress/plot/<exp_name>')
def plot(exp_name):
    return static_file('plots.png', root=os.path.join(EXP_ROOT, exp_name, 'progress'))

@route('/css')
def css():
    return static_file('style.css', root=SCRIPT_DIR)

run(host='fstop.eecs.umich.edu', port=80, debug=True)

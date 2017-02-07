import os
from bottle import route, run, static_file, template

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ROOT = os.path.join(SCRIPT_DIR, '..', '..', 'experiments')

def is_running(exp_name):
    return os.path.exists(os.path.join(EXP_ROOT, exp_name, 'RUNNING'))

@route('/')
def root():
    ret = ''
    for exp_name in sorted(os.listdir(EXP_ROOT)):
        exp_text = exp_name + (' RUNNING' if is_running(exp_name) else '')
        exp_tag = '<a href="/progress/%s">%s</a>' % (exp_name, exp_text)
        ret = ret + exp_tag + '<br>'
    return ret

@route('/progress/<exp_name>')
def progress(exp_name):
    exp_full_path = os.path.join(EXP_ROOT, exp_name)
    with open(os.path.join(exp_full_path, 'README.md'), 'r') as f:
        readme_contents = f.read()
    readme_contents = readme_contents.replace(os.linesep, '<br>')

    # Add progress plot, if it exists
    if os.path.exists(os.path.join(exp_full_path, 'progress', 'plots.png')):
        image_tag = '<img width="100%%" src="/progress/plot/%s" />' % exp_name
    else:
        image_tag = ''

    return readme_contents + '<br>' + image_tag

@route('/progress/plot/<exp_name>')
def plot(exp_name):
    return static_file('plots.png', root=os.path.join(EXP_ROOT, exp_name, 'progress'))

run(host='localhost', port=8080, debug=True)

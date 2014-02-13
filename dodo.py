
import os

from doit import get_var
from doit.action import CmdAction

figure_extension = '.pdf'

### Process arguments
def _get_boolean_argument(name, default=None):
    value = get_var(name, default)
    if default is None and value is None:
        raise ValueError("Required argument '%s' not provided" % name)

    value = str(value).lower()
    if value in ['1', 'true', 'yes', 'y', True]:
        return True
    if value in ['0', 'false', 'no', 'n', False]:
        return False
    raise ValueError(
        "Unrecognized value '%s' for parameter '%s'" % (value, name))


gpu = _get_boolean_argument('gpu', False)
if gpu:
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'


### Helper functions
def _result(task, model):
    return os.path.join('results', '%s_%s.npz' % (task, model))


def _figure(number, task):
    return os.path.join(
        'figures', 'figure%d_%s%s' % (number, task, figure_extension))


def _simulate(task, func, model):
    name = _result(task, model)
    return {'basename': 'sim_%s_%s' % (task, model),
            'actions': [(func, [model, name])],
            # 'file_dep': ['scripts/%s.py' % task],
            'uptodate': [lambda task, values: True],
            'targets': [name],
            'verbosity': 2,
            }


def _plot(num, task, func, deps):
    target = _figure(num, task)
    return {'basename': 'plot_%s' % (task),
            'actions': [(func, [target])],
            'file_dep': deps + [os.path.join('scripts', 'makefigures.py')],
            'targets': [target],
            'doc': "Generate figure %d: %s" % (num, task),
            'clean': True,
            }


### Tasks
def task_sim_info():
    """Run simulations for information plots"""
    def sim_info(model, save_name):
        import scripts.info
        scripts.info.run(model, save_name)

    simulate = lambda m: _simulate('info', sim_info, m)
    yield simulate('lif')
    yield simulate('fhn')


def task_plot_info():
    deps = [_result('info', 'lif'), _result('info', 'fhn')]

    def plot_infohetero(target):
        import scripts.makefigures
        scripts.makefigures.infoheteroplot(*deps, target=target)
    yield _plot(2, 'infohetero', plot_infohetero, deps)

    def plot_infocontour(target):
        import scripts.makefigures
        scripts.makefigures.infocontourplot(*deps, target=target)
    yield _plot(3, 'infocontour', plot_infocontour, deps)

    def plot_infonoise(target):
        import scripts.makefigures
        scripts.makefigures.infonoiseplot(*deps, target=target)
    yield _plot(4, 'infonoise', plot_infonoise, deps)


def task_sim_phase():
    """Run simulations for phase plot"""
    def sim_phase(model, save_name):
        import scripts.phase
        scripts.phase.run(model, save_name)

    simulate = lambda m: _simulate('phase', sim_phase, m)
    yield simulate('lif')
    yield simulate('fhn')


def task_plot_phase():
    deps = [_result('phase', 'lif'), _result('phase', 'fhn')]

    def plot_phase(target):
        import scripts.makefigures
        scripts.makefigures.phaseplot(*deps, target=target)

    return _plot(5, 'phase', plot_phase, deps)


def task_sim_firing():
    """Run simulations for firing (spike raster) plot"""
    def sim_firing(model, save_name):
        import scripts.firing
        scripts.firing.run(model, save_name)

    return _simulate('firing', sim_firing, 'lif')


def task_plot_firing():
    deps = [_result('firing', 'lif')]

    def plot_firing(target):
        import scripts.makefigures
        scripts.makefigures.syncrasterplot(*deps, target=target)

    return _plot(6, 'syncraster', plot_firing, deps)


def task_sim_tuning():
    """Run simulations for tuning plot"""
    def sim_tuning(model, save_name):
        import scripts.tuning
        scripts.tuning.run(model, save_name)

    yield _simulate('tuning', sim_tuning, 'lif')
    yield _simulate('tuning', sim_tuning, 'fhn')


def task_plot_tuning():
    deps = [_result('tuning', 'lif'), _result('tuning', 'fhn')]

    def plot_tuning(target):
        import scripts.makefigures
        scripts.makefigures.tuningnoisyplot(*deps, target=target)

    return _plot(7, 'tuningnoisy', plot_tuning, deps)


def task_plot_tuninghetero():
    def plot_tuninghetero(target):
        import scripts.makefigures
        scripts.makefigures.tuningheteroplot(target=target)

    return _plot(8, 'tuninghetero', plot_tuninghetero, [])


def task_paper():
    """Generate a PDF of the paper using pdflatex"""
    command = ""
    if not _get_boolean_argument('figures', True):
        command += "\\newcommand{\\hidefigures}{\\relax}"
    command += "\\input{manuscript.tex}"

    pdf = CmdAction(
        'pdflatex -interaction=batchmode "%s"' % command, cwd='paper')
    return {'actions': [pdf, pdf, pdf],
            'file_dep': [os.path.join('paper', 'manuscript.tex')],
            'targets': [os.path.join('paper', 'manuscript.pdf')],
            'clean': True}

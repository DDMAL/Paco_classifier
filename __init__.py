import rodan
__version__ = rodan.__version__

import logging
logger = logging.getLogger('rodan')

from rodan.jobs import module_loader

module_loader('rodan.jobs.Calvo-classifier.calvo_classifier')
module_loader('rodan.jobs.Calvo-classifier.calvo_trainer')
#module_loader('rodan.jobs.Calvo-classifier.fast_calvo_classifier')


from keras.callbacks import Callback as Callback
import keras.backend as K
import numpy as np
from utils import compute_del_grauman

class GradientReversalScheduler(Callback):
	def __init__(self, l, schedule, verbose=1):
		self.schedule = schedule
		self.verbose = verbose
		self.l = l

	def on_epoch_begin(self, epoch, logs={}):
		l = self.schedule(epoch)
		#K.set_value(self.l, l)
		K.get_session().run(self.l.assign(value=l))
		if self.verbose > 0:
			print('\nEpoch %05d: GradientReversalScheduler increases lambda '
				  'to %s.' % (epoch + 1, l))

# More work needed here.
class L21DelWeightsUpdate(Callback):
	def __init__(self, del_grauman, predicate_groups, predicates):
		self.del_grauman = del_grauman
		self.predicate_groups = predicate_groups
		self.predicates = predicates

	def on_epoch_begin(self, epoch, logs={}):
		wmat = self.model.layers[-1].get_weights()[0]
		self.del_grauman = compute_del_grauman(wmat, self.predicate_groups, self.predicates)
		print 'Del_grauman updated'

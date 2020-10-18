#Author - Saugat Kandel
# coding: utf-8

from .curveball import Curveball, PreconditionedCurveball
from .lma import LMA
from .merit_lma import MeritLMA
from .nlcg import NonLinearConjugateGradient
from .projected_gradient import ProjectedGradient
from .utils import *
from .utils import AdaptiveLineSearch, BackTrackingLineSearch
from .utils import MatrixFreeLinearOp, conjugate_gradient

#__all__ = ["Curveball", "LMA", "NonLinearConjugateGradient", "PreconditionedCurveball", "MeritLMA",
#           "AdaptiveLineSearch", "BackTrackingLineSearch", "MatrixFreeLinearOp", "conjugate_gradient"]

__all__ = ['AdaptiveLineSearch',
           'BackTrackingLineSearch',
           'MatrixFreeLinearOp',
           'conjugate_gradient',
           'NonLinearConjugateGradient',
           'ProjectedGradient',
           'Curveball',
           'PreconditionedCurveball',
           'LMA',
           'MeritLMA']
#__all__ += utils.__all__
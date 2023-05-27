from causalGraph import CausalGraph
from FCIFunctions import deleteSeparatedEdges, addColliders, deleteSeparatedEdgesStage2, zhangsInferenceRules, completeUndirectedEdges, completeDirectedEdges, jci, background
from CDCFunctions import CDCrules
from independenceTest import IndependenceTest
from collections import defaultdict
import copy


def FCI(variables, data=None, corr=None, data_size=-1, alpha=0.05, indTest='Fisher', conservative_FCI=False, 
    selection_bias=False, oracle_SS=None, JCI='0', context=[], CDC=False, beta=0.05, conservative_CDC=False, 
    oracle_CDC=None, bkg_info=None, aug_PC=False, verbose=False):
    """
    Returns a causalGraph object that represents a (over/undercomplete) PAG.
        :param variables: list of the non-context variables of the system (strings)
        :param data: dictionary (keys:variables/context) of np-arrays
        :param data: correlation matrix
        :param data_size: integer representing the number of data points, default is -1
        :param alpha: threshold for independence for m-separation, default is 0.05
        :param indTest: string describing independence test, default is 'Fisher',
                         and 'kernel' is other option 
        :param conservative_FCI: boolean indicating if we run conservative FCI
        :param selection_bias: boolean indicating if we run selection bias, only possible 
                        if JCI='0' and CDC=False
        :param oracle_SS: dictionary (keys:variables) of dictionary (keys:variables) for
                        separating sets -> {v1:{v2:[ss1, ss2]}} means v1 and v2 are
                        separated by set ss1 and separated by set ss2 (default is None)
        :param JCI: string ('0', '1', '12', '123', default is '0') indicating which  
                        assumtions from 'Joint Causal Inference from Multiple Contexts'
                        by Joris Mooij et al. (2020) are assumed to be true
        :param context: list of the context variables of the system (strings)
        :param CDC: boolean indicating if we run CDC
        :param beta: threshold for independence for CDC, default is 0.05
        :param conservative_CDC: boolean indicating if we run conservative CDC
        :param oracle_CDC: causalGraph without variant edges
        :param bkg_info: background information in the form of an ancestral graph 
                        represented by a matrix where [i,j]=1 if i is an ancestor of j, 
                        [i,j]=1 if i is a non-descendants of j, and 0 means no info
        :param aug_PC: boolean indicating if we run augmented PC instead of FCI 
                        (skip D in FCI from Causation Prediction and Search (p.145))
        :param verbose: boolean whether to print information during the run
    """

    if oracle_SS is None:
        if data is None and not corr is None and indTest!='Fisher':
            indTest = 'Fisher'
            print('Independence test is set to Fisher-Z test since data set was not avaiable.')
        elif data is None and corr is None:
            raise ValueError('Provide either a data set, correlation matrix or oracle_SS information.')
        elif not data is None and corr is None and indTest=='Fisher':
            data_ = {}
            for var in variables:
                data_[var] = list(data[var][:,0])
            for var in context:
                data_[var] = list(data[var][:,0])
            data_ = pd.DataFrame.from_dict(data_)
            corr = data_.corr(method='spearman') # monotonic relationship
            data_size = data[[*data][0]].shape[0]

    # define independence test object
    IT = IndependenceTest(alpha=alpha, indTest=indTest, oracle=oracle_SS)

    # check if selection bias is possible
    if selection_bias:
        if (len(context)>0 and JCI!='0') or CDC:
            selection_bias = False
            print('Selection bias is set to false due to conflicting parameters.')

    # check if background information is consistent with context
    if not bkg_info is None and len(context)>0:
        for X in context:
            if -1 in bkg_info.loc[X, :].unique() or 1 in bkg_info.loc[:, X].unique():
                raise ValueError('The background information is not in line with the context variables.')

    # if JCI is not used: add context variables to other variables
    if JCI == '0':
        if len(context) > 0:
            variables = variables + context
            context = []

    if verbose:
        print('initialize causal graph')

    # define a complete undirected graph
    pag = CausalGraph(variables, data=data, corr=corr, data_size=data_size, unknown='u', default_edge='u', context=context)
    separating_sets = defaultdict(list)
    uncertain_triples = defaultdict(list)

    if verbose:
        print('delete separated edges')

    # delete edges between variables which are d-separated based on sets in oracle
    deleteSeparatedEdges(pag, separating_sets, JCI=JCI, IT=IT, oracle=oracle_SS)

    if verbose:
        print('orient context variables')

    # orient edges according to JCI algorithm
    jci(pag, JCI=JCI)


    if verbose:
        print('orient background information')

    # add background information based on ancestral graph
    background(pag, bkg_info=bkg_info)

    if not aug_PC and oracle_SS is None:

        if verbose:
            print('delete separated edges: stage 2')

        # add colliders: R0
        pag_copy = copy.deepcopy(pag)
        addColliders(pag_copy, separating_sets, conservative=conservative_FCI, IT=IT, uncertain_triples=uncertain_triples)

        # search for separated vertices and delete edges
        deleteSeparatedEdgesStage2(pag, pag_copy, separating_sets, IT)

        # reset uncertain triples
        uncertain_triples = defaultdict(list)

    if verbose:
        print('orient colliders')

    # add colliders: R0
    addColliders(pag, separating_sets, conservative=conservative_FCI, IT=IT, uncertain_triples=uncertain_triples)

    if verbose:
        print('run rule R1-R4')

    # Zhangs version of Meeks orientation rules: R1-R4
    zhangsInferenceRules(pag, separating_sets, uncertain_triples=uncertain_triples, IT=IT, conservative=conservative_FCI)


    # additional orientation rules if there is selection bias
    if selection_bias:

        if verbose:
            print('run rule R5-R7')

        completeUndirectedEdges(pag)

    if verbose:
        print('run rule R8-R10')

    # complete directed edges: R8-R10
    completeDirectedEdges(pag)

    if CDC:

        if verbose:
            print('run CDC')

        # CDC orientation
        CDCrules(pag, separating_sets, oracle=oracle_CDC, beta=beta, conservative=conservative_CDC)

    return pag


import numpy as np

############################################
# variables to encode amino-acids as numbers
aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
aadict = {aa[k]:k for k in range(len(aa))}
############################################


def seq2num(S):
    if type(S) == str:
        return [aadict[x] for x in S]
    elif type(S) == list:
        return np.array([[aadict[x] for x in s] for s in S])


def _compute_profile_aligned(seqs, verbose=False):
    """
    Given the set of sequences `seqs` (list of strings having all the same lengths), 
    returns a position weight matrix (profile). Sequences can be composed of the 20 standard amino-acides, 
    and of gaps ('-' symbols).
    """
    #checks
    assert type(seqs) == list, "Sequences must be provided as list of strings."
    assert type(seqs[0]) == str, "Sequences must be provided as list of strings."
    assert len(np.unique([len(s) for s in seqs])) == 1, "Sequences must have all the same length."

    # compute seq2num
    if verbose:
        print("[compute_profile] Computing profile (seq2num)...", end='')
    seqs_2num = seq2num(seqs)
    if verbose:
        print(" done!")
    
    # compute profile
    if verbose:
        print("[compute_profile] Computing profile...", end='')
    profile = np.full((seqs_2num.shape[1], len(aa)), 0, dtype=np.float64)
    for i in range(profile.shape[0]):
        nums_pseudo = np.full(len(aa), 1)
        vals, obs = np.unique(seqs_2num[:, i], return_counts=True)
        nums = np.copy(nums_pseudo)
        for v,n in zip(vals, obs):
            nums[v] += n
        nums = nums / np.sum(nums)
        profile[i,:] = np.log(nums)
    if verbose:
        print(" done!")
    return profile


def align_profiles(prof1, prof2, N1, N2):
    """
    Given two profiles whose lengths differ by 1, add a column in the shortest profile so that
    the other columns are aligned with maximum score. The score is the sum of the overlaps of 
    probability vectors of the two profiles, for each column. Then a new profile is build by 
    computing the new probabilities (taking the gaps into account) and taking the log. `N1` 
    and `N2` are the number of sequences used to build, respectively `prof1` and `prof2`, and 
    are used to weight the two profiles when creating the new one. 
    
    Return a profile and its new number of sequences (`N1` plus `N2`).
    """
    # checks
    assert np.abs(prof1.shape[0] - prof2.shape[0]) == 1, "Profiles must have difference in length equal to 1."
    
    # find short and long profile
    prof_s = np.exp([prof1, prof2][np.argmin([prof1.shape[0], prof2.shape[0]])])
    prof_l = np.exp([prof1, prof2][np.argmax([prof1.shape[0], prof2.shape[0]])])
    N_s = [N1, N2][np.argmin([prof1.shape[0], prof2.shape[0]])]
    N_l = [N1, N2][np.argmax([prof1.shape[0], prof2.shape[0]])]
    
    # prepare a "gap-only" state
    gap_ins = np.full(prof_l.shape[1], 1/(N_s + prof_l.shape[1]))
    gap_ins[-1] = N_s/(N_s + prof_l.shape[1])
    
    # find best gap position    
    scores = []    
    for gp in range(prof_l.shape[0]):    
        pre = sum([np.dot(prof_s[i], prof_l[i]) for i in range(gp)]) 
        post = sum([np.dot(prof_s[i], prof_l[i+1]) for i in range(gp, prof_l.shape[0]-1)])
        gap = np.dot(gap_ins, prof_l[gp])
        scores.append(pre+post+gap)
    gp_best = np.argmax(scores)
    
    # prepare gappy profile from the short one
    prof_sl = np.zeros_like(prof_l)
    prof_sl[:gp_best, :] = prof_s[:gp_best, :]
    prof_sl[gp_best, :] = gap_ins
    prof_sl[gp_best+1:, :] = prof_s[gp_best:, :]
    
    # fuse the two profiles and take the log
    totN = N_s + N_l
    w_s = N_s / totN
    w_l = N_l / totN
    out_prof = np.log( w_s * prof_sl + w_l * prof_l )
    return out_prof, totN


def seq_profile_score(seq, profile):
    """
    Compute the loglikelihood of the sequence `seq` given the profile.
    """
    assert type(seq) == str, "Seq must be a string."
    assert len(seq) == profile.shape[0], "Seq length and profile length do not correspond."
    s2n = seq2num(seq)
    return sum([profile[i,x] for i,x in enumerate(s2n)])
    

def seq_align_to_profile(seq, profile):
    """
    Given the sequence `seq` and the profile `profile`, return a new sequence aligned to the profile, as follows:
        - if the length of `seq` is the same as of `profile`, return `seq`;
        - if `seq` is shorter than `profile`, add a single stretch of gaps so to maximize the
            loglikelihood of the sequence given the profile;
        - if `seq` is shorter than `profile`, delete a single stretch of amino-acids so to maximize the
            loglikelihood of the sequence given the profile.
    Return the aligned sequence.
    """
    L = len(seq)
    prof_L = profile.shape[0]
    if L < prof_L:
        l_gaps = prof_L - L
        s_gap = "-" * l_gaps
        scores = []
        for p in range(L):
            scores.append(seq_profile_score(seq[:p] + s_gap + seq[p:], profile))
        p_best = np.argmax(scores)
        return seq[:p_best] + s_gap + seq[p_best:]
    elif L == prof_L:
        return seq
    else:
        l_cut = L - prof_L
        scores = []
        for p in range(L - l_cut+1):
            scores.append(seq_profile_score(seq[:p] + seq[p+l_cut:], profile))
        p_best = np.argmax(scores)
        return seq[:p_best] + seq[p_best+l_cut:]


def seqs_align_to_profile(seqs, profile, verbose=False):
    """
    Run `seq_align_to_profile` to each sequence of `seqs`, and return the result.
    """
    if verbose:
        print("[align_to_profile] Aligning...")
    return [seq_align_to_profile(seq, profile) for seq in seqs]


def build_profile_from_seqs(seqs, L_final=20, Lmin=8):
    """
    Build a profile from sequence `seqs`, building profiles of sequences of fixed length starting from the maximum between 
    the minimum length in `seqs and `Lmin`, and aligning to the sequences of higher length through `align_profiles`.
    Then the profile is trimmed by discarding the most gappy columns until the lenght `L_final` is obtained.
    """
    assert L_final <= max([len(x) for x in seqs]), "`L_final` must be not larger than the maximum length in `seqs`."
    L = max(min([len(x) for x in seqs]), Lmin)
    cdr3b_L = [x for x in seqs if len(x) == L]

    t_prof = _compute_profile_aligned(cdr3b_L)
    t_N = len(cdr3b_L)

    # align to profiles of increasing lengths
    for L in range(max(min([len(x) for x in seqs]), Lmin)+1, max([len(x) for x in seqs])+1):
        cdr3b_L = [x for x in seqs if len(x) == L]
        prof_L = _compute_profile_aligned(cdr3b_L)
        t_prof, t_N = align_profiles(prof_L, t_prof, len(cdr3b_L), t_N)

    # build final profile by trimming gappy columns
    prof_final = t_prof[np.sort(np.argsort(t_prof[:, -1])[:L_final])]

    p_Cfirst = np.exp(prof_final[0][1])
    p_Flast = np.exp(prof_final[-1][4])
    if (p_Cfirst < 0.9) or (p_Flast < 0.9):
        print("WARNING: from the final profile, these do not seem CDR3-beta sequences. ", end='')
        print("If indeed they are not, please notice that this function has been on CDR3-beta sequences only, and consider using another aligner. ", end='')
        print("If they are CDR3-beta sequences, be cautious: probably something went wrong with the alignment.")
    return prof_final


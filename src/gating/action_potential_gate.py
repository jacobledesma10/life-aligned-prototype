class ActionPotentialGate:
    def __init__(self, necessity_thresh=0.4, alignment_thresh=0.6, risk_thresh=0.3):
        self.necessity_thresh = necessity_thresh
        self.alignment_thresh = alignment_thresh
        self.risk_thresh = risk_thresh

    def allow_action(self, necessity, alignment, risk):
        if necessity > self.necessity_thresh and \
           alignment > self.alignment_thresh and \
           risk < self.risk_thresh:
            return True
        return False

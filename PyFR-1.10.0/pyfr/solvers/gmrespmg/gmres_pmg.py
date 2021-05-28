import tioga as tg

class gmres_pmg(tg.gmrespmg):
    def __init__(self, rhsupdate, pmgupdate):
        tg.gmrespmg.__init__(self)

        # also initialize the 
        self.rhsupdate = rhsupdate
        self.pmgupdate = pmgupdate

    def pmgupdate(self):
        pass

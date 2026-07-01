class SeismicTraceOutput:
    """
    Generic class for seismic traces output

    Attributes
    -----------
        data : array-like
            seismic traces to export
        format : str
            Output format \
            "SEP" or "SEGY"
    """
    def __init__(self, seismo, format, **kwargs):
        """
        Parameters
        -----------
            seismo : array-like
                Seismic traces to export
            format : str
                Output format \
                "SEP" or "SEGY"
        """
        self.data = seismo
        self.format = format


    def export(self, **kwargs):
        """
        Save the seismic traces in the requested format
        """
        if self.format.lower() == "sep":
            from .SEPTraceOutput import SEPTraceOutput
            seismoOut = SEPTraceOutput(self.data, **kwargs)
            seismoOut.export(**kwargs)
        
        elif self.format.lower() == "segy":
            from .SEGYTraceOutput import SEGYTraceOutput
            seismoOut = SEGYTraceOutput(self.data, **kwargs)
            seismoOut.export(**kwargs)
        
        else:
            raise TypeError("Unknown output format")

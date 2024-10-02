import logging


log_formatter = logging.Formatter(
    fmt='%(asctime)s %(name)s %(message)s',
)


class ListHandler(logging.Handler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loglist = []

    def handle(self, record):
        self.loglist += [record]
        # print('Handling this record:', formatter.format(record))

    def get_str_formatting(self, log_list=None):
        if log_list is None:
            list_to_use = self.loglist
        else:
            list_to_use = log_list

        out_str = []

        for record in list_to_use:
            out_str += [log_formatter.format(record)]
        return '\n'.join(out_str)

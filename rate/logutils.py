import logging
import tqdm

### From https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
###
### Usage:
###
###
# import time
#
# log = logging.getLogger (__name__)
# log.setLevel (logging.INFO)
# log.addHandler (TqdmLoggingHandler ())
# for i in tqdm.tqdm (range (100)):
#     if i == 50:
#         log.info ("Half-way there!")
#     time.sleep (0.1)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
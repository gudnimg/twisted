twisted.internet.defer.Deferred will no longer remove the traceback object from Failures. This may result in more objects staying in memory if you don't clean up failed Dferreds, but it will speed up error handling and enable improvements to traceback reporting.
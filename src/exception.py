import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    """Constructs a detailed error message for logging.

    This function extracts the file name, line number, and the error message 
    from the exception details, and formats them into a single string.

    Args:
        error (Exception): The exception that was raised.
        error_detail (sys): The sys module, used to access traceback information.

    Returns:
        str: A formatted error message containing the script name, line number, 
             and the original error message.

    Example:
        try:
             1 / 0
         except Exception as e:
             error_message = error_message_detail(e, sys)
    """    
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    """Custom exception class to handle and log errors.

    This class extends the built-in Exception class to provide additional context
    for errors that occur in the application. It captures detailed information
    about the error, including the location in the code where it occurred.

    Attributes:
        error_message (str): A formatted message detailing the error, including 
                             the script name and line number.
    """    

    def __init__(self,error_message,error_detail:sys):
        """Initializes the CustomException with a message and error details.

        Args:
            error_message (str): The error message provided during the exception.
            error_detail (sys): The sys module, used to access traceback information.
        """
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        """Returns the error message as a string.

        Returns:
            str: The formatted error message for the exception.
        """ 
        return self.error_message
    

# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         # logging.info('Division by zero occurred')
#         raise CustomException(e,sys)
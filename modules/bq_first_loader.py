
#import datetime
from google.cloud import bigquery
import pandas
import pytz

class BQFirstLoader():
    def __init__(self):
        # Construct a BigQuery client object.
        print("Connecting to BQ..... ", end="")
        self._client = bigquery.Client(project='jardim-data')
        
        self._job_config = bigquery.LoadJobConfig(    
            write_disposition="WRITE_TRUNCATE"
        )
        print(" - Connected!")


    def load_df_to_table(self, dataframe, table_name):
        table_id = "jardim-data.dl_tutoring." + table_name

        # Make an API request.
        job = self._client.load_table_from_dataframe(
            dataframe, table_id, job_config=self._job_config
        )  
        job.result()  # Wait for the job to complete.

        # Make an API request.
        table = self._client.get_table(table_id)  
        print(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), table_id
            )
        )
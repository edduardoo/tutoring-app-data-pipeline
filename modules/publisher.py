from google.cloud import pubsub_v1
import json

class Publisher():
    
    def __init__(self, topic):
        project_id = "jardim-data"        
        print("Connecting to PubSub.... ", end="")
        self._client = pubsub_v1.PublisherClient()
        # creates a fully qualified identifier in the form `projects/{project_id}/topics/{topic_id}`
        print(" - Connected!")
        self._topic_path = self._client.topic_path(project_id, topic)
        
        
    def publish_df_row(self, df_row):        
        data = df_row.to_json(date_format='iso')        
        data = data.encode("utf-8")
                
        future = self._client.publish(self._topic_path, data)
        #return future
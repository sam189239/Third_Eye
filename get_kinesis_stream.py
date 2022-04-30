import boto3
import cv2
from botocore.config import Config

# my_config = Config(
#     region_name = 'us-west-2',
#     signature_version = 'v4',
#     retries = {
#         'max_attempts': 10,
#         'mode': 'standard'
#     }
# )



STREAM_NAME = "5ZY4HSUIYKHTPFPN"
# stream_arn = "arn:aws:kinesisvideo:us-west-2:255845548276:stream/VP52M8OQZ10HOQRB/1636963448487"
kvs = boto3.client("kinesisvideo" )


# response = kvs.list_signaling_channels(
#     MaxResults=123,
#     # NextToken='string',
#     ChannelNameCondition={
#         'ComparisonOperator': 'BEGINS_WITH',
#         'ComparisonValue': 'Stream'
#     }
# )

# print(response)

# Grab the endpoint from GetDataEndpoint
endpoint = kvs.get_data_endpoint(
    APIName="GET_HLS_STREAMING_SESSION_URL",
    StreamName=STREAM_NAME,
    # StreamARN = stream_arn
    )['DataEndpoint']


print(endpoint)


# # Grab the HLS Stream URL from the endpoint
kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)
url = kvam.get_hls_streaming_session_url(
    StreamName=STREAM_NAME,
    # PlaybackMode="ON_DEMAND",
    PlaybackMode="LIVE"
    )['HLSStreamingSessionURL']


print(url)


vcap = cv2.VideoCapture(url)


while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()


    if frame is not None:
        # Display the resulting frame
        cv2.putText(frame,"success",(32,32),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
        cv2.imshow('frame',frame)



        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

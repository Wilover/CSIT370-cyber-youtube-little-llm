import pandas as pd
import requests
import json
import googleapiclient.discovery

dev = "YOUR_DEVELOPER_KEY_HERE"  # Replace with your actual YouTube Data API v3 developer key

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = dev

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)


def getcomments(video):
  request = youtube.commentThreads().list(
      part="snippet",
      videoId=video,
      maxResults=100
  )



  comments = []

  # Execute the request.
  response = request.execute()

  # Get the comments from the response.
  for item in response['items']:
      comment = item['snippet']['topLevelComment']['snippet']
      public = item['snippet']['isPublic']
      comments.append([
          comment['authorDisplayName'],
          comment['publishedAt'],
          comment['likeCount'],
          comment['textOriginal'],
          comment['videoId'],
          public
      ])

  while (1 == 1):
    try:
     nextPageToken = response['nextPageToken']
    except KeyError:
     break
    nextPageToken = response['nextPageToken']
    # Create a new request object with the next page token.
    nextRequest = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100, pageToken=nextPageToken)
    # Execute the next request.
    response = nextRequest.execute()
    # Get the comments from the next response.
    for item in response['items']:
      comment = item['snippet']['topLevelComment']['snippet']
      public = item['snippet']['isPublic']
      comments.append([
          comment['authorDisplayName'],
          comment['publishedAt'],
          comment['likeCount'],
          comment['textOriginal'],
          comment['videoId'],
          public
      ])

  df2 = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','video_id','public'])
  return df2


video_ids = [
    "d8d9EZHU7fw",
    "GK9MESMpR7s",
    "umyoCyZCkyg",
    "XNMHUifKce8",
    "MaFK5AXpXXw",
    "aFa1ike_Q7I",
    "2jU-mLMV8Vw",
    "MmdwakT-Ve8",
    "IS4OgH74gY4",
    "Gr4GdROJIZ8",
    "m5t08CREHcE",
    "CMHL1bPtQdI",
    "-qGAM0Mt2V0",
    "iyPIR8pLksQ",
    "oY0RdRTW2YY",
    "WHOgdsEiyew",
    "3yiT_WMlosg",
    "4t4kBkMsDbQ",
    "5xWnmUEi1Qw",
    "QGC40AfmgY0",
    "VtFaQjTcRts"
]

long_comment_videos = [
#     "XNMHUifKce8",
#     "MaFK5AXpXXw",
#     "2jU-mLMV8Vw",
#     "MmdwakT-Ve8",
#     "IS4OgH74gY4",
#     "m5t08CREHcE",
#     "iyPIR8pLksQ"
]

df = pd.DataFrame()

for video_id in video_ids:
    df2 = getcomments(video_id)
    if video_id in long_comment_videos:
        df2 = df2[df2["text"].str.len() >= 60]

    df = pd.concat([df, df2], ignore_index=True)

df['video_id'].value_counts()
df.to_csv("data/youtube_comments_cs_dataset.csv", index=False, encoding="utf-8-sig")
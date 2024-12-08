import requests
import json
from bs4 import BeautifulSoup
from lxml import etree
import ffmpeg.video
import os
import re
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip


# Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0

class BilibiliVideoAudio:
    def __init__(self, bid,l,r):
        self.bid = bid
        self.l = l
        self.r = r
        self.headers = {
            "referer": "https://www.bilibili.com",
            "origin": "https://www.bilibili.com",
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
            'Accept-Encoding': 'identity'
        }

    def get_video_audio(self):
        ans = []
        # 构造视频链接并发送请求获取页面内容
        # url = f'https://www.bilibili.com/video/{self.bid}?spm_id_from=333.851.b_7265636f6d6d656e64.6'
        for i in range(self.l,self.r):

            url = f'https://www.bilibili.com/video/{self.bid}?p={i}'
            content = requests.get(url, headers=self.headers).content.decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')

            # 获取视频标题
            meta_tag = soup.head.find('meta', attrs={'name': 'title'})
            title = meta_tag['content']

            # 获取视频和音频链接
            pattern = r'window\.__playinfo__=({.*?})\s*</script>'
            json_data = re.findall(pattern, content)[0]
            data = json.loads(json_data)

            video_url = data['data']['dash']['video'][0]['base_url']
            audio_url = data['data']['dash']['audio'][0]['base_url']

            ans.append({
                'title': title,
                'video_url': video_url,
                'audio_url': audio_url
            })

        # return {
        #     'title': title,
        #     'video_url': video_url,
        #     'audio_url': audio_url
        # }
        return ans

    def download_video_audio(self, url, filename):
        # 对文件名进行清理，去除不合规字符
        filename = self.sanitize_filename(filename)
        try:
            # 发送请求下载视频或音频文件
            resp = requests.get(url, headers=self.headers).content
            download_path = os.path.join('download', filename)  # 构造下载路径
            with open(download_path, mode='wb') as file:
                file.write(resp)
            print("{:*^30}".format(f"下载完成：{filename}"))
        except Exception as e:
            print(e)

    def sanitize_filename(self, filename):
        # 定义不合规字符的正则表达式
        invalid_chars_regex = r'[\"*<>?\\|/:,]'

        # 替换不合规字符为空格
        sanitized_filename = re.sub(invalid_chars_regex, ' ', filename)

        return sanitized_filename

    def merge(self,title):
        video = VideoFileClip(f'download/{title}.mp4')
        audio = AudioFileClip(f'download/{title}.mp3')
        video_merge = video.set_audio(audio)
        output_path = "download"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = f"{output_path}/{title}_merged.mp4"
        video_merge.write_videofile(output_file)
        os.remove(f'download/{title}.mp4')
        os.remove(f'download/{title}.mp3')

def main(bids,l,r):
    # bids = ["BV1Rw411P7MX"]  # 视频的bid，可以修改为其他视频的bid
    for bid in bids:
        bilibili = BilibiliVideoAudio(bid,l,r + 1)
        video_audio_infos = bilibili.get_video_audio()

        for video_audio_info in video_audio_infos:
            title = video_audio_info['title']
            video_url = video_audio_info['video_url']
            audio_url = video_audio_info['audio_url']

            bilibili.download_video_audio(video_url, f"{title}.mp4")  # 下载视频
            bilibili.download_video_audio(audio_url, f"{title}.mp3")  # 下载音频

            bilibili.merge(title)

# bids = []
# name = input("输入需要爬取的视频bid:")
# while name != '0':
#     bids.append(name)
#     name = input()
bids = ['BV1d54y1g7db']

l, r = map(int, input("需要需要爬取的集数：").split())
main(bids=bids,l = l,r = r)
#
# # BV1py4y1E7oA
# # BV1d54y1g7db
# # BV1d54y1g7db


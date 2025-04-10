import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import random

def get_hznu_news():
    # 新闻网站URL列表
    urls = [
        "https://www.hznu.edu.cn/xww/",  # 新闻网首页
        "https://www.hznu.edu.cn/xww/xydt/",  # 校园动态
        "https://www.hznu.edu.cn/xww/xxyw/",  # 学校要闻
        "https://www.hznu.edu.cn/xww/mtbd/"   # 媒体杭师
    ]
    
    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1'
    }
    
    news_data = {
        "news_list": [],
        "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        print("正在获取新闻数据...")
        session = requests.Session()
        
        # 先访问主页
        session.get("https://www.hznu.edu.cn/", headers=headers, timeout=10)
        
        for url in urls:
            print(f"\n正在抓取: {url}")
            # 添加随机延时
            time.sleep(random.uniform(1, 3))
            
            try:
                response = session.get(url, headers=headers, timeout=10)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找所有可能的新闻容器
                news_containers = soup.find_all(['div', 'li'], class_=lambda x: x and ('news' in str(x).lower() or 'list' in str(x).lower()))
                
                for container in news_containers:
                    try:
                        # 在容器中查找链接
                        links = container.find_all('a')
                        for link in links:
                            title_text = link.get_text().strip()
                            href = link.get('href', '')
                            
                            # 过滤无效的链接和标题
                            if (title_text and len(title_text) > 5 and
                                not title_text.startswith('http') and
                                not any(char.isdigit() for char in title_text[:2])):
                                
                                # 处理相对链接
                                if href and not href.startswith('http'):
                                    if href.startswith('/'):
                                        href = 'https://www.hznu.edu.cn' + href
                                    else:
                                        href = url.rstrip('/') + '/' + href.lstrip('/')
                                
                                news_item = {
                                    "title": title_text,
                                    "url": href,
                                    "source": url.split('/')[-2] if url.split('/')[-2] else "首页"
                                }
                                
                                # 检查是否已经存在相同标题
                                if not any(item["title"] == title_text for item in news_data["news_list"]):
                                    news_data["news_list"].append(news_item)
                    except Exception as e:
                        print(f"处理新闻容器时出错: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"抓取页面 {url} 时出错: {str(e)}")
                continue
        
        # 按标题长度排序，通常较长的标题更可能是新闻标题
        news_data["news_list"].sort(key=lambda x: len(x["title"]), reverse=True)
        
        print(f"\n找到 {len(news_data['news_list'])} 条新闻")
        
        # 保存数据到JSON文件
        with open('hznu_news.json', 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
        print("\n数据已保存到 hznu_news.json")
        
        # 打印部分数据预览
        print("\n=== 数据预览 ===")
        print(f"总共获取到 {len(news_data['news_list'])} 条新闻标题")
        print("\n前10条新闻标题:")
        for i, news in enumerate(news_data["news_list"][:10], 1):
            print(f"\n{i}. {news['title']}")
            print(f"   链接: {news['url']}")
            print(f"   来源: {news['source']}")
        
        # 将数据转换为纯文本格式保存
        with open('hznu_news.txt', 'w', encoding='utf-8') as f:
            f.write(f"杭州师范大学新闻列表 (爬取时间: {news_data['crawl_time']})\n\n")
            for i, news in enumerate(news_data["news_list"], 1):
                f.write(f"{i}. {news['title']}\n")
                f.write(f"   链接: {news['url']}\n")
                f.write(f"   来源: {news['source']}\n\n")
        print("\n数据已同时保存到 hznu_news.txt")
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    get_hznu_news()


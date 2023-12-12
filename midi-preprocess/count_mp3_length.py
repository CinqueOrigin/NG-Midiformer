from mutagen.mp3 import MP3
import argparse
import os


def traverse_mp3_dir(events_dir):
    """
    从events文件夹中获取所有的events文件名称
    :param events_dir:
    :return:
    """
    file_names = []
    for f in os.listdir(events_dir):
        if f.endswith(".mp3"):
            file_names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    return file_names


def read_lrc(file_path, encoding="gb18030"):
    with open(file_path, encoding=encoding) as fin:
        text = fin.readlines()

    res = []
    remain = ''
    for line in text:
        line = line.split(']')
        if len(line) != 2:
            continue
        line = line[1].strip()
        if len(line) > 0:
            if len(remain) < 32:
                remain += ' ' + line
            else:
                res.append(remain)
                remain = ''

    if len(remain) > 0:
        res.append(remain)

    return res


def main():
    parser = argparse.ArgumentParser(description='Count mp3 total length.')
    parser.add_argument('--mp3_path', help='mp3 path')
    args = parser.parse_args()

    mp3_files = traverse_mp3_dir(args.mp3_path)

    fc = 0
    ts = 0
    ss = 0
    for f in mp3_files:
        if os.path.exists(os.path.join(args.mp3_path, f + ".lrc")):
            audio = MP3(os.path.join(args.mp3_path, f + ".mp3"))
            content = read_lrc(os.path.join(args.mp3_path, f + ".lrc"))
            fc += 1
            ts += audio.info.length
            ss += len(content)
    print('total: ' + str(fc))
    print('sum: ' + str(ts) + 's')
    print('sentence sum: ' + str(ss))


if __name__ == '__main__':
    main()

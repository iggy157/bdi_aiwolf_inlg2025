#!/usr/bin/env python3
import os
import yaml
import glob
from pathlib import Path


def load_mbti_data(filepath):
    """MBTIデータを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def load_enneagram_data(filepath):
    """エニアグラムデータを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def calculate_desire_tendencies(mbti_data, enneagram_data):
    """願望傾向を計算する"""
    
    # MBTIから各指標を取得
    extroversion = mbti_data.get('extroversion', 0)
    introversion = mbti_data.get('introversion', 0)
    sensing = mbti_data.get('sensing', 0)
    intuition = mbti_data.get('intuition', 0)
    
    # エニアグラムから各指標を取得
    reformer = enneagram_data.get('reformer', 0)
    achiever = enneagram_data.get('achiever', 0)
    peacemaker = enneagram_data.get('peacemaker', 0)
    
    # 願望傾向の計算
    desire_tendencies = {
        '自己実現': intuition * 0.6 + reformer * 0.4,
        '社会的承認': sensing * 0.5 + achiever * 0.5,
        '安定性': introversion * 0.6 + peacemaker * 0.4,
        '愛情親密さ': introversion * 0.5 + peacemaker * 0.5,
        '自由独立': extroversion * 0.7 + reformer * 0.3,
        '冒険刺激': extroversion * 0.6 + intuition * 0.4,
        '安定的な人間関係': introversion * 0.6 + peacemaker * 0.4
    }
    
    return desire_tendencies


def save_desire_tendencies(desire_tendencies, output_path):
    """願望傾向をYMLファイルに保存する"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(desire_tendencies, f, allow_unicode=True, default_flow_style=False)


def process_all_agents():
    """すべてのエージェントの願望傾向を計算して保存する"""
    base_path = "./info/bdi_info/macro_bdi/macro_belief"
    
    # すべてのMBTIファイルを検索
    mbti_files = glob.glob(f"{base_path}/*/*/mbti.yml")
    
    print(f"検索パス: {base_path}/*/*/mbti.yml")
    print(f"見つかったMBTIファイル数: {len(mbti_files)}")
    
    for mbti_file in mbti_files:
        # 対応するエニアグラムファイルのパスを構築
        enneagram_file = mbti_file.replace('mbti.yml', 'enneagram.yml')
        
        if not os.path.exists(enneagram_file):
            print(f"警告: エニアグラムファイルが見つかりません: {enneagram_file}")
            continue
        
        try:
            # データを読み込み
            mbti_data = load_mbti_data(mbti_file)
            enneagram_data = load_enneagram_data(enneagram_file)
            
            # 願望傾向を計算
            desire_tendencies = calculate_desire_tendencies(mbti_data, enneagram_data)
            
            # 出力パスを構築
            output_path = mbti_file.replace('mbti.yml', 'desire_tendencies.yml')
            
            # 結果を保存
            save_desire_tendencies(desire_tendencies, output_path)
            
            print(f"処理完了: {output_path}")
            
        except Exception as e:
            print(f"エラー: {mbti_file} の処理中にエラーが発生しました: {e}")


def main():
    """メイン関数"""
    print("願望傾向計算を開始します...")
    process_all_agents()
    print("願望傾向計算が完了しました。")


if __name__ == "__main__":
    main()
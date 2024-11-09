from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import torch
import faiss
import pickle
import pandas as pd
import os
import gdown

class EmbeddingModel:
    def __init__(self, model_name='jinaai/jina-colbert-v2', cache_folder='./model_cache'):
        """تهيئة نموذج Embedding"""
        self.embedding_model = None
        self.index = None
        self.poetry_data = []
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._initialize_model()

    def _initialize_model(self):
        """تهيئة النموذج"""
        try:
            print("جاري تحميل نموذج Embedding...")
            self.embedding_model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                cache_folder=self.cache_folder
            )
            print("تم تحميل النموذج بنجاح!")
        except Exception as e:
            print(f"خطأ في تحميل النموذج: {str(e)}")
            raise e

    def _create_embeddings(self, texts):
        """إنشاء Embeddings للنصوص مع إدارة الذاكرة"""
        try:
            print(f"جاري معالجة {len(texts)} نص...")
            batch_size = 8  # تقليل حجم الدفعة
            embeddings_list = []

            for i in tqdm(range(0, len(texts), batch_size), desc="معالجة النصوص"):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        batch_size=1
                    )

                    batch_embeddings = np.array(batch_embeddings, dtype='float32')
                    embeddings_list.append(batch_embeddings)

                    if i % (batch_size * 100) == 0 and i > 0:
                        print(f"\nتم معالجة {i} نص")
                        temp_embeddings = np.vstack(embeddings_list)
                        np.save(f'temp_embeddings_{i}.npy', temp_embeddings)
                        embeddings_list = []
                        gc.collect()

                except Exception as batch_error:
                    print(f"خطأ في معالجة الدفعة {i}: {str(batch_error)}")
                    continue

            print("\nجاري دمج النتائج...")
            final_embeddings_list = []
            temp_files = [f for f in os.listdir('.') if f.startswith('temp_embeddings_')]

            for temp_file in temp_files:
                temp_embeddings = np.load(temp_file)
                final_embeddings_list.append(temp_embeddings)
                os.remove(temp_file)

            if embeddings_list:
                final_embeddings_list.append(np.vstack(embeddings_list))

            embeddings = np.vstack(final_embeddings_list)
            print(f"اكتملت المعالجة! شكل المصفوفة النهائية: {embeddings.shape}")

            return embeddings

        except Exception as e:
            print(f"خطأ في إنشاء Embeddings: {str(e)}")
            raise e

    def process_poetry_data(self, file_path='poetry_data.csv'):
        """معالجة ملف البيانات الشعرية"""
        try:
            print("قراءة ملف البيانات...")
            df = pd.read_csv(file_path, sep=',', encoding='utf-8')

            print("أعمدة الملف:", df.columns.tolist())
            print(f"عدد الأبيات: {len(df)}")

            self.poetry_data = df.to_dict('records')

            print("\nجاري إنشاء Embeddings...")
            texts = df['poem_text'].tolist()
            print(f"تم تحميل {len(texts)} نص")

            embeddings = self._create_embeddings(texts)

            print("\nجاري بناء الفهرس...")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)

            print("\nجاري حفظ البيانات...")
            faiss.write_index(self.index, 'embeddings.faiss')
            with open('poetry_data.pkl', 'wb') as f:
                pickle.dump(self.poetry_data, f)

            print("اكتملت المعالجة بنجاح!")
            return True

        except Exception as e:
            print(f"خطأ في معالجة ملف البيانات: {str(e)}")
            print("\nمحتوى الملف:")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f.readline())
            except Exception as read_error:
                print(f"خطأ في قراءة الملف: {str(read_error)}")
            return False

def test_model():
    try:
        print("إنشاء النموذج...")
        model = EmbeddingModel()


        print("\nمعالجة البيانات...")
        success = model.process_poetry_data(output)

        if not success:
            print("فشل في معالجة البيانات!")
            return

    except Exception as e:
        print(f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    test_model()

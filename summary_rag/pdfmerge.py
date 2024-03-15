from PIL import Image
import img2pdf
import os

def jpgs_to_pdf(jpg_files, output_pdf):
    with open(output_pdf, "wb") as pdf_file:
        pdf_file.write(img2pdf.convert(jpg_files))

if __name__ == '__main__':
    output_pdf = '/home/vqa/masterthesis/ourproject/summary_rag/output2.pdf'
    directory = "/home/vqa/masterthesis/ourproject/summary_rag/OPPO_WATCH_User_Manual/images/"
    jpg_files = [os.path.join(directory, file) for file in sorted(os.listdir(directory)) if file.lower().endswith('.jpg')]
    print("JPG files:", jpg_files)
    jpgs_to_pdf(jpg_files, output_pdf)

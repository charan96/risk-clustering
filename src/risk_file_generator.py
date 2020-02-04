import os
import re
import glob
from bs4 import BeautifulSoup as bs


class RiskFileGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def create_risk_files_dir_if_not_exists(self):
        dir_path = os.path.join(self.data_dir, 'risk_files')
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        return dir_path
    
    def create_risk_filename(self, raw_filename):
        basename = raw_filename.split('/')[-2]
        
        risk_files_dir_path = self.create_risk_files_dir_if_not_exists()
        
        return os.path.join(risk_files_dir_path, basename + '.risk')
        
    def prettify_html_text(self, html_text):
        soup = bs(html_text)
        pretty_html = soup.prettify()
        
        return pretty_html
    
    def preprocess_raw_file_data(self, raw_data):
        pretty_html = self.prettify_html_text(raw_data)
        
        pretty_html = re.sub(r'<.*?>', ' ', pretty_html)
        pretty_html = re.sub(r'&.*?;', ' ', pretty_html)
        pretty_html = re.sub(r'[ \t]+', ' ', pretty_html)
        pretty_html = re.sub(r'\xa0', ' ', pretty_html)
        
        clean_html = [line.strip() for line in pretty_html.split("\n") if line != " " and line != "\t" and line != ""]
        
        return "\n".join(clean_html)
    
    def extract_risk_factors_section(self, preprocessed_data):
        matches = re.findall(r'\nI[t|T][e|E][m|M][\s\n]*1[a|A]\.?[\s\n]*R[i|I][s|S][k|K][\s\n]*F[a|A][c|C][t|T][o|O][r|R][s|S][\s\n]*((.|\n)*?)[\s\n]*I[t|T][e|E][m|M][\s\n]*1[b|B]', preprocessed_data)
        
        risk_factors_section = '\n'.join(matches[-1])
        risk_factors_section = risk_factors_section.replace("\n", " ")

        return risk_factors_section
    
    def generate_risk_file(self, raw_filename):
        out_filename = self.create_risk_filename(raw_filename)
        
        with open(raw_filename, 'r') as in_fh, open(out_filename, 'w') as out_fh:
            raw_data = in_fh.readlines()
            raw_data = ''.join(raw_data)
            
            preprocessed_data = self.preprocess_raw_file_data(raw_data)
            risk_factors_section = self.extract_risk_factors_section(preprocessed_data)
            
            out_fh.write(risk_factors_section)

        return out_filename

    def get_risk_filename_from_raw_file(self, raw_filename):
        return self.create_risk_filename(raw_filename)
            

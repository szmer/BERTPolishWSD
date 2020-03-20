import http.client
import urllib.parse

class NetworkError(Exception):
    pass

def krnnt_response_lines(text):
    """
    Get KRNNT's analysis of text as a list of lines.
    """
    params = urllib.parse.urlencode({'text': text})
    headers = {'Content-type': 'application/x-www-form-urlencoded',
            'Accept': 'text/plain'}
    conn = http.client.HTTPConnection('localhost:9003') # where KRNNT resides
    conn.request('POST', '', params, headers)
    response = conn.getresponse()
    if response.status != 200:
        raise NetworkError('Cannot connect to KRNNT: {} {}, is the container running?'.format(
            response.status, response.reason))
    resp_html = response.read().decode('utf-8') # we get a HTML page and need to strip tags
    lines = resp_html[resp_html.index('<pre>')+len('<pre>')
            :resp_html.index('</pre>')].strip().split('\n')
    return lines

def tag_nkjp(text):
    """
    Tag text with NKJP tagset. Returns a list of sentences as lists of (form, lemma, interp).
    """
    lines = krnnt_response_lines(text)
    sents = [[]]
    current_token = None
    for line_n, line in enumerate(lines):
        # Next sentence.
        if len(line) == 0:
            sents.append([])
            continue
        # A form line - assign the form.
        if line_n % 2 == (0 if (len(sents) % 2 == 1) else 1):
            current_token = line.split('\t')[0]
        else:
            interp_data = line.split('\t')
            current_token = (current_token, interp_data[1], interp_data[2])
            sents[-1].append(current_token)
    sents = [sent for sent in sents if len(sent) > 0]
    return sents

def split_sents_krnnt(text, strip_sents=True):
    """
    Use KRNNT to split the text into a list of sentence strings.
    """
    lines = krnnt_response_lines(text)
    sents = ['']
    for line_n, line in enumerate(lines):
        # Next sentence.
        if len(line) == 0:
            sents.append('')
            continue
        # A form line - grow the sentence.
        if line_n % 2 == (0 if (len(sents) % 2 == 1) else 1):
            form_data = line.split('\t')
            form = form_data[0]
            if form_data[1] == 'space':
                preceding_sep = ' '
            elif form_data[1] == 'none':
                preceding_sep = ''
            elif form_data[1] == 'newline':
                preceding_sep = '\n'
            sents[-1] += preceding_sep + form
    if strip_sents:
        sents = [sent.strip() for sent in sents]
    sents = [sent for sent in sents if len(sent) > 0]
    return sents

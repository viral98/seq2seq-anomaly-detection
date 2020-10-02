# THIS IS NO LONGER MAINTAINED
# seq2seq-anomaly-detection
A Natural Language Processing based approach to detect malicious HTTP requests. Originally forked from [seq2seq-web-attack-detection](https://github.com/PositiveTechnologies/seq2seq-web-attack-detection) .

## Model Parameters
• batch_size - 128 - the number of samples in a batch.

• embed_size - 64 -  the dimension of embedding space (should be less than vocabulary size).

• hidden_size - 64 -the number of hidden states in lstm.

• num_layers - 2 - the number of lstm blocks.

• checkpoints - path to checkpoint directory.

• std_factor - 6. - the number of stds that is used for defining a model threshold.

• dropout - 0.7 - the probability that each element is kept.

• vocab - the Vocabulary object.

In the parameters setting stage,  the threshold setting is introduced. Set_threshold calculates the threshold value using mean and std of loss values of valid samples.

At the testing stage, the model receives benign and anomalous samples. For each sample, the value of loss is calculated. If this value is greater than the threshold, then the request is considered anomalous.

## Sample Data

### Benign Sample

GET /vulnbank/assets/fonts/Pe-icon-7-stroke.woff?d7yf1v HTTP/1.1

Host: 10.0.212.25

Connection: keep-alive

Origin: http://10.0.212.25

User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36

Accept: */*

Referer: http://10.0.212.25/vulnbank/assets/css/pe-icon-7-stroke.css

Accept-Encoding: gzip, deflate

Accept-Language: en-US,en;q=0.9

Cookie: PHPSESSID=j1pavglp5ue30266c0j88ged30

### Anomalous sample

POST /vulnbank/online/api.php HTTP/1.1

Host: 10.0.212.25

User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:59.0) Gecko/20100101 Firefox/59.0

Accept: application/json, text/javascript, */*; q=0.01

Accept-Language: en-US,en;q=0.5

Accept-Encoding: gzip, deflate

Referer: http://10.0.212.25/vulnbank/online/login.php

Content-Type: application/x-www-form-urlencoded; charset=UTF-8

X-Requested-With: XMLHttpRequest

Content-Length: 209

Cookie: PHPSESSID=mlacs0uiou344i3fa53s7raut6

Connection: keep-alive

type=user&action=create&username=Jack'+and+extractvalue(0x0a,concat(0x0a,(select version())))+and+'1'='1&password=passw0rd&firstname=first&lastname=last&birthdate=30-08-2017&email=eee%40mail.com&phone=747474747&account=DE44404419569750553340&creditcard=4556-9373-3913-6510


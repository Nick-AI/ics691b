# Reading

## Amazon Redshift and Aurora

* Aurora (standard transactional database, SQL) -> Redshift (big data analysis, data warehouse)
    1) Once data is stored in DB (aurora) use Lambda function to autom. transfer (subset) to S3 "bucket" (large storage location)
        * Lambda function actually wraps kinesis"data delivery stream" that triggers when relevant data is added/ changed
    2) Now when data stored in aurora it also gets sent to S3
    3) Can use Amazon redshift spectrum from a connected redshift cluster to query any kind of data stored in S3 with no additional processing
* Once in redshift, can use different amazon applications to analyze/ visualize data
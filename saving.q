q)t
sym    src price
--------------------
EURUSD ebs 1.058962
EURUSD ebs 1.38322
EURUSD rtr 0.4593231
EURUSD rtr 1.383906

Tenum:.Q.en[`:db]t

Savedown:{`:/data/01/hdb/2017.07.09/t/ set select from t where src=`ebs; `:/data/02/hdb/2017.07.09/t/ set select from t where src=`rtr}
`:/db/par.txt 0: (“/data/01/hdb/”;”/data/02/hdb/”)

OR
`:/db/par.txt 0: (“/data/01/hdb/”;”/data/02/hdb/”;”/data/03/hdb/”;”/data/04/hdb/”)

q).cfg.par
ebs| “:/data/01/hdb/” “:/data/02/hdb/"
rtr| “:/data/03/hdb/” “:/data/04/hdb/"

savedown:{[dt]
                      ebspar: .cfg.par[`ebs]( dt mod 2);
                       rtrpar: .cfg.par[`rtr] (dt mod 2);      // here 2 can be replaced by n depending on the number of segments you want to have for that src
                      (`$(.cfg.par[`ebs] (.z.d mod 2)),string[dt],"/" ) set select from tenum where src=`ebs;
                      (`$(.cfg.par[`rtr] (.z.d mod 2)),string[dt],"/" ) set select from tenum where src=`ebs;

                       }

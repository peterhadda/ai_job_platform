package model;

public class Job {
    string jobId;
    string title;
    string company;
    string location;
    string datePosted;
    string description;
    string url;
    string source;

    public Job(string jobId, string title,string company,string location,string datePosted,string description,string url,string source){
        this.jobId=jobId;
        this.title=title;
        this.company=company;
        this.location=location;
        this.datePosted=datePosted;
        this.description=description;
        this.url=url;
        this.source=source;
    }

    public string getJobId(){
        return jobId;
    }
    public string getTitle(){
        return title;
    }
    





}

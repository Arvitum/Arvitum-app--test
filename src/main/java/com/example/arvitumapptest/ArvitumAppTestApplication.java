package com.example.arvitumapptest;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;

@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class })
public class ArvitumAppTestApplication {

    public static void main(String[] args) {
        int A = 0;
        String mynewstring = "helloworld";
        SpringApplication.run(ArvitumAppTestApplication.class, args);
    }
//todo: some test here
    
}

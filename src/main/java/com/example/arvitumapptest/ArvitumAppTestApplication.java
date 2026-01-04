package com.example.arvitumapptest;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.ai.tool.annotation.ToolParam;


import java.util.ArrayList;
import java.util.List;

@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class })
public class ArvitumAppTestApplication {

    public static void main(String[] args) {
        List<String> stringlist = new ArrayList<>();
        SpringApplication.run(ArvitumAppTestApplication.class, args);
    }

    //todo: make some changes here for test
}

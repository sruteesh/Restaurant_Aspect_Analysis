
<!DOCTYPE html>
<!--
To change this license header, choose License Headers in Project Properties.
To change this template file, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE html>
<html dir="ltr" lang="en-US">
    <head>

         <meta http-equiv="content-type" content="text/html; charset=utf-8" />
        <meta name="Lubna" content="Official Website" />
        
        <!-- Stylesheets
        ============================================= -->
        <link href="http://fonts.googleapis.com/css?family=Lato:300,400,400italic,600,700|Raleway:300,400,500,600,700|Crete+Round:400italic" rel="stylesheet" type="text/css" />
        
        <link rel="stylesheet" href="style.css" type="text/css" />
        <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>

        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <!--[if lt IE 9]>
            <script src="http://css3-mediaqueries-js.googlecode.com/svn/trunk/css3-mediaqueries.js"></script>
        <![endif]-->

        <!-- External JavaScripts
        ============================================= -->
      
        
        
        
        
        <!-- Document Title
        ============================================= -->
        <title>drive: Actionable Insights from Visual Data</title>

    </head>

    <body class="stretched">

        <!-- Document Wrapper
        ============================================= -->
        
         <script type="text/javascript">
       $(document).ready(function () {
             $("#submit").on('submit', function (e) {
             e.preventDefault();
              var name = $("#content").val();
              var datastring = "content="+name;
              console.log($(this).serialize());
              
        $.ajax({
            type: 'POST',
            url: 'https://dev.neuronme.com/demo/api/sentimentanalysis/',
            data: $(this).serialize(),
            success: function (data) {
              
                $('#results').html(data);
                //$('#submit').hide();
            }
        });
    });
});
    </script>
       

            
            <section id='con'>
            
                <div class='api_content'>
                
                    <div class="api_wrapper">
                        
                        <div class='api_intro'>
                            <h4>Text Analysis Demo</h4>
                            <p>Enter some text in the box given below to explore sentiment analysis API</p>
                        </div>
                        
                        <div class='api_input_wrapper'>
                            <form id="submit" >
                            
                                <div class="api_input">
                                <textarea id="content"  placeholder="Please enter some text" required="" class=""></textarea>
                                
                                </div>
                                <div class="api_submit">
                                <button type="submit" class="api_submit_btn" id="api_submit_btn">See Results</button>
                                </div>
                            </form>
                        </div>
                                                
                    </div>
                </div>
           
            </section> 
        
            <div id="results">Results will go here</div>


        <!-- Footer Scripts
        ============================================= -->
       

    </body>
</html>
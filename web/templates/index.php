<!DOCTYPE html>
<html dir="ltr" lang="en-US">
    <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8" />
        <meta name="" content="Official Website" />
        
        <!-- Stylesheets
        ============================================= -->
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}" type="text/css" />
        <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>

        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <!-- External JavaScripts
        ============================================= -->
      
        
        <!-- Document Title
        ============================================= -->
        <title>drive: Actionable Insights from Texts</title>

    </head>

    <body class="stretched">

        <!-- Document Wrapper
        ============================================= -->
        
        <script type="text/javascript">
            // if(isset($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH'])=='xmlhttprequest')
            //     {
            //     echo "Ajax Request";
            //     }
            //     else
            //     {
            //     echo "Invalid Request";
            //     }
            $(document).ready(function () {
                $("#submit").on('submit', function (e) {
                $("button#api_submit_btn").prop('disabled', true)
                $("button#api_submit_btn").text('Please Wait...');
                e.preventDefault();
                var text_1 = $("#text").val();
                // var domain_1 = $("#submit").val();
                var domain = $("input[name='domain']:checked").val();
                var datastring = "text="+text_1 + "&domain="+ domain;
                console.log(datastring);
                console.log(domain)
                console.log($(this).serialize());
        $.ajax({
            type: 'POST',
            url: '/textapi/',
            dataType: 'json',
            data: datastring,
            success: function (data) {
                var returnedData = JSON.stringify(data, null, 2);
                setTimeout(function() {
                    console.log(returnedData)
                    $("button#api_submit_btn").prop('disabled',false);
                    $("button#api_submit_btn").text('See Results');
                    $("textarea#results").val(returnedData);
                    }, 3000);
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
                        <p>Enter some text in the box given below and select the text api to explore text analysis API</p>
                    </div>
                    
                    <div class='api_input_wrapper'>
                        <form id="submit" >
                            <input type="radio" name="domain" value="sentiment" checked> Sentiment
                            <input type="radio" name="domain" value="hotels"> Hotel Review
                            <input type="radio" name="domain" value="restaurants"> Restaurant Review
                            <br>
                            <div class="api_input">
                                <textarea id="text"  placeholder="Please enter some text" required="" class=""></textarea>
                            </div>
                            <div class="api_submit">
                                <button id="api_submit_btn" type="submit"  >See Results</button>
                            </div>
                        </form>
                    </div>                                         
                </div>
            </div>
        </section> 
        <form>
            <!-- <div id="results">Results will go here</div> -->
            <textarea id="results"  placeholder="Results will go here" rows="20" cols="500" disabled="disabled" style="width:1500px;"></textarea>
            <!-- <textarea id="results"  placeholder="Results will go here" required="" class=""></textarea> -->
        </form>
        <!-- Footer Scripts
        ============================================= -->
    </body>
</html>
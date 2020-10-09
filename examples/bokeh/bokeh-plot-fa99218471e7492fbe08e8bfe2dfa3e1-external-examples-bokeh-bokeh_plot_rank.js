(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("6f79c3f9-a9b2-4b36-bfeb-aa2b9e8fcf05");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6f79c3f9-a9b2-4b36-bfeb-aa2b9e8fcf05' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"7df06a37-ffbc-432b-b99c-b1e443f9af7b":{"roots":{"references":[{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26664","type":"VBar"},{"attributes":{"data_source":{"id":"26657"},"glyph":{"id":"26658"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26659"},"selection_glyph":null,"view":{"id":"26661"}},"id":"26660","type":"GlyphRenderer"},{"attributes":{"source":{"id":"26657"}},"id":"26661","type":"CDSView"},{"attributes":{},"id":"26711","type":"Selection"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26725"},"ticker":{"id":"26703"}},"id":"26628","type":"LinearAxis"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26665","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26662","type":"Span"},{"attributes":{},"id":"26712","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26715"},"selection_policy":{"id":"26716"}},"id":"26663","type":"ColumnDataSource"},{"attributes":{},"id":"26632","type":"ResetTool"},{"attributes":{"source":{"id":"26663"}},"id":"26667","type":"CDSView"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26671","type":"VBar"},{"attributes":{},"id":"26584","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26640","type":"BoxAnnotation"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26670","type":"VBar"},{"attributes":{"children":[[{"id":"26581"},0,0],[{"id":"26617"},0,1]]},"id":"26737","type":"GridBox"},{"attributes":{"data_source":{"id":"26663"},"glyph":{"id":"26664"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26665"},"selection_glyph":null,"view":{"id":"26667"}},"id":"26666","type":"GlyphRenderer"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26708"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26591"}},"id":"26590","type":"LinearAxis"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26668","type":"Span"},{"attributes":{},"id":"26622","type":"LinearScale"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26717"},"selection_policy":{"id":"26718"}},"id":"26669","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26669"}},"id":"26673","type":"CDSView"},{"attributes":{"ticks":[0,1,2,3]},"id":"26675","type":"FixedTicker"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26723"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26625"}},"id":"26624","type":"LinearAxis"},{"attributes":{},"id":"26723","type":"BasicTickFormatter"},{"attributes":{"ticks":[0,1,2,3]},"id":"26703","type":"FixedTicker"},{"attributes":{},"id":"26713","type":"Selection"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26684","type":"Span"},{"attributes":{"data_source":{"id":"26669"},"glyph":{"id":"26670"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26671"},"selection_glyph":null,"view":{"id":"26673"}},"id":"26672","type":"GlyphRenderer"},{"attributes":{},"id":"26725","type":"BasicTickFormatter"},{"attributes":{},"id":"26714","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26674","type":"Span"},{"attributes":{},"id":"26586","type":"LinearScale"},{"attributes":{"data_source":{"id":"26679"},"glyph":{"id":"26680"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26681"},"selection_glyph":null,"view":{"id":"26683"}},"id":"26682","type":"GlyphRenderer"},{"attributes":{},"id":"26582","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26681","type":"VBar"},{"attributes":{},"id":"26637","type":"UndoTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26680","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26726"},"selection_policy":{"id":"26727"}},"id":"26679","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"26590"}],"center":[{"id":"26593"},{"id":"26597"},{"id":"26656"},{"id":"26662"},{"id":"26668"},{"id":"26674"}],"left":[{"id":"26594"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26654"},{"id":"26660"},{"id":"26666"},{"id":"26672"}],"title":{"id":"26677"},"toolbar":{"id":"26608"},"toolbar_location":null,"x_range":{"id":"26582"},"x_scale":{"id":"26586"},"y_range":{"id":"26584"},"y_scale":{"id":"26588"}},"id":"26581","subtype":"Figure","type":"Plot"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26687","type":"VBar"},{"attributes":{},"id":"26638","type":"SaveTool"},{"attributes":{"toolbars":[{"id":"26608"},{"id":"26642"}],"tools":[{"id":"26598"},{"id":"26599"},{"id":"26600"},{"id":"26601"},{"id":"26602"},{"id":"26603"},{"id":"26604"},{"id":"26605"},{"id":"26632"},{"id":"26633"},{"id":"26634"},{"id":"26635"},{"id":"26636"},{"id":"26637"},{"id":"26638"},{"id":"26639"}]},"id":"26738","type":"ProxyToolbar"},{"attributes":{"source":{"id":"26679"}},"id":"26683","type":"CDSView"},{"attributes":{},"id":"26715","type":"Selection"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26728"},"selection_policy":{"id":"26729"}},"id":"26685","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"26624"}],"center":[{"id":"26627"},{"id":"26631"},{"id":"26684"},{"id":"26690"},{"id":"26696"},{"id":"26702"}],"left":[{"id":"26628"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26682"},{"id":"26688"},{"id":"26694"},{"id":"26700"}],"title":{"id":"26705"},"toolbar":{"id":"26642"},"toolbar_location":null,"x_range":{"id":"26582"},"x_scale":{"id":"26620"},"y_range":{"id":"26584"},"y_scale":{"id":"26622"}},"id":"26617","subtype":"Figure","type":"Plot"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26686","type":"VBar"},{"attributes":{},"id":"26716","type":"UnionRenderers"},{"attributes":{"text":"mu"},"id":"26705","type":"Title"},{"attributes":{"source":{"id":"26685"}},"id":"26689","type":"CDSView"},{"attributes":{},"id":"26633","type":"PanTool"},{"attributes":{},"id":"26620","type":"LinearScale"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26641","type":"PolyAnnotation"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26652","type":"VBar"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26693","type":"VBar"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26692","type":"VBar"},{"attributes":{"data_source":{"id":"26685"},"glyph":{"id":"26686"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26687"},"selection_glyph":null,"view":{"id":"26689"}},"id":"26688","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26690","type":"Span"},{"attributes":{"children":[{"id":"26739"},{"id":"26737"}]},"id":"26740","type":"Column"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26730"},"selection_policy":{"id":"26731"}},"id":"26691","type":"ColumnDataSource"},{"attributes":{},"id":"26726","type":"Selection"},{"attributes":{"source":{"id":"26691"}},"id":"26695","type":"CDSView"},{"attributes":{"text":"tau"},"id":"26677","type":"Title"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26699","type":"VBar"},{"attributes":{},"id":"26727","type":"UnionRenderers"},{"attributes":{},"id":"26635","type":"WheelZoomTool"},{"attributes":{},"id":"26717","type":"Selection"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26698","type":"VBar"},{"attributes":{},"id":"26598","type":"ResetTool"},{"attributes":{"callback":null},"id":"26639","type":"HoverTool"},{"attributes":{"data_source":{"id":"26691"},"glyph":{"id":"26692"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26693"},"selection_glyph":null,"view":{"id":"26695"}},"id":"26694","type":"GlyphRenderer"},{"attributes":{},"id":"26718","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26696","type":"Span"},{"attributes":{"axis":{"id":"26594"},"dimension":1,"ticker":null},"id":"26597","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26732"},"selection_policy":{"id":"26733"}},"id":"26697","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"26738"},"toolbar_location":"above"},"id":"26739","type":"ToolbarBox"},{"attributes":{"source":{"id":"26697"}},"id":"26701","type":"CDSView"},{"attributes":{"overlay":{"id":"26640"}},"id":"26634","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"26697"},"glyph":{"id":"26698"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26699"},"selection_glyph":null,"view":{"id":"26701"}},"id":"26700","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"26641"}},"id":"26636","type":"LassoSelectTool"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26702","type":"Span"},{"attributes":{},"id":"26728","type":"Selection"},{"attributes":{},"id":"26729","type":"UnionRenderers"},{"attributes":{},"id":"26599","type":"PanTool"},{"attributes":{"overlay":{"id":"26607"}},"id":"26602","type":"LassoSelectTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26606","type":"BoxAnnotation"},{"attributes":{},"id":"26591","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26653","type":"VBar"},{"attributes":{},"id":"26708","type":"BasicTickFormatter"},{"attributes":{},"id":"26603","type":"UndoTool"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26711"},"selection_policy":{"id":"26712"}},"id":"26651","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"26624"},"ticker":null},"id":"26627","type":"Grid"},{"attributes":{"callback":null},"id":"26605","type":"HoverTool"},{"attributes":{},"id":"26710","type":"BasicTickFormatter"},{"attributes":{},"id":"26730","type":"Selection"},{"attributes":{"overlay":{"id":"26606"}},"id":"26600","type":"BoxZoomTool"},{"attributes":{"axis":{"id":"26628"},"dimension":1,"ticker":null},"id":"26631","type":"Grid"},{"attributes":{"source":{"id":"26651"}},"id":"26655","type":"CDSView"},{"attributes":{},"id":"26731","type":"UnionRenderers"},{"attributes":{},"id":"26625","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26598"},{"id":"26599"},{"id":"26600"},{"id":"26601"},{"id":"26602"},{"id":"26603"},{"id":"26604"},{"id":"26605"}]},"id":"26608","type":"Toolbar"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26632"},{"id":"26633"},{"id":"26634"},{"id":"26635"},{"id":"26636"},{"id":"26637"},{"id":"26638"},{"id":"26639"}]},"id":"26642","type":"Toolbar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26713"},"selection_policy":{"id":"26714"}},"id":"26657","type":"ColumnDataSource"},{"attributes":{},"id":"26604","type":"SaveTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26607","type":"PolyAnnotation"},{"attributes":{},"id":"26601","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26656","type":"Span"},{"attributes":{"data_source":{"id":"26651"},"glyph":{"id":"26652"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26653"},"selection_glyph":null,"view":{"id":"26655"}},"id":"26654","type":"GlyphRenderer"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26710"},"ticker":{"id":"26675"}},"id":"26594","type":"LinearAxis"},{"attributes":{"axis":{"id":"26590"},"ticker":null},"id":"26593","type":"Grid"},{"attributes":{},"id":"26732","type":"Selection"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26658","type":"VBar"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26659","type":"VBar"},{"attributes":{},"id":"26733","type":"UnionRenderers"},{"attributes":{},"id":"26588","type":"LinearScale"}],"root_ids":["26740"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"7df06a37-ffbc-432b-b99c-b1e443f9af7b","root_ids":["26740"],"roots":{"26740":"6f79c3f9-a9b2-4b36-bfeb-aa2b9e8fcf05"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();